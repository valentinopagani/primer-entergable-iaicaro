import csv
import unicodedata
from pathlib import Path
from typing import Optional
from langchain.tools import tool
from pydantic import BaseModel, Field

from mesa_pedidos import reglas

CSV_PRODUCTOS = Path(__file__).resolve().parent.parent / "datos" / "catalogo.csv"
CSV_CLIENTES = Path(__file__).resolve().parent.parent / "datos" / "clientes.csv"


def _leer_csv(ruta: Path) -> list[dict]:
    with open(ruta, encoding="utf-8-sig") as csvfile:
        return list(csv.DictReader(csvfile))


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFD", (t or "").lower()).encode("ascii", "ignore").decode()
    return " ".join(t.split())


def buscar_producto(consulta: str) -> dict | None:
    """Compat legacy: SKU exacto o substring en nombre."""
    consulta = (consulta or "").strip().upper()
    if not consulta:
        return None
    for p in _leer_csv(CSV_PRODUCTOS):
        if p["sku"].upper() == consulta or consulta in p["nombre"].upper():
            return p
    return None


def buscar_productos_por_texto(descripcion: str, limite: int = 3) -> list[dict]:
    """Candidatos por coincidencia de palabras (para sugerencias)."""
    toks = [t for t in _norm(descripcion).split() if len(t) > 2]
    if not toks:
        return []
    scored = []
    for p in _leer_csv(CSV_PRODUCTOS):
        nombre = _norm(p["nombre"])
        score = sum(1 for t in toks if t in nombre)
        if score:
            scored.append((score, p))
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:limite]]


def buscar_cliente(cuit: str) -> dict | None:
    if not cuit:
        return None
    cuit = cuit.replace("-", "").strip()
    for c in _leer_csv(CSV_CLIENTES):
        if c["cuit"].replace("-", "").strip() == cuit:
            return c
    return None


class ConsultarProductoInput(BaseModel):
    sku: Optional[str] = Field(default=None, description="Codigo XX-0000 si se conoce")
    descripcion: Optional[str] = Field(default=None, description="Descripcion libre si no hay SKU")
    cantidad: int = Field(default=1, ge=1, description="Cantidad a comprar")
    sku_o_nombre: Optional[str] = Field(default=None, description="Legacy: SKU o nombre en un solo campo")


class VerificarCreditoInput(BaseModel):
    cuit: str = Field(description="CUIT con o sin guiones")
    monto_pedido_neto: Optional[float] = Field(default=None, description="Legacy")
    monto_neto: Optional[float] = Field(default=None, description="Neto sin IVA con descuentos")
    pago_anticipado: bool = Field(default=False)


class CotizarFleteInput(BaseModel):
    provincia: str = Field(description="Provincia destino")
    localidad: str = Field(default="", description="Localidad destino")
    monto_neto: float = Field(ge=0)
    pago_anticipado: bool = Field(default=False)
    unidades_aerosol: int = Field(default=0, ge=0)


def _producto_a_dict(p: dict) -> dict:
    return {"sku": p["sku"], "nombre": p["nombre"], "rubro": p["rubro"],
            "marca": p["marca"], "unidad": p["unidad"],
            "precio_lista": float(p["precio_lista"]), "alicuota_iva": float(p["alicuota_iva"]),
            "stock": int(p["stock"]), "deposito": p["deposito"]}


@tool(args_schema=ConsultarProductoInput)
def consultar_producto(sku: Optional[str] = None, descripcion: Optional[str] = None,
                       cantidad: int = 1, sku_o_nombre: Optional[str] = None) -> dict:
    """Busca en el catalogo real por SKU o descripcion. Informa, no decide."""
    if sku_o_nombre and not sku and not descripcion:
        q = sku_o_nombre.strip()
        import re
        if re.match(r"(?i)^[a-z]{2}-?\d{3,4}$", q.replace(" ", "")):
            sku = q.upper()
        else:
            descripcion = q
    if sku:
        s = sku.strip().upper()
        for p in _leer_csv(CSV_PRODUCTOS):
            if p["sku"].strip().upper() == s:
                d = _producto_a_dict(p)
                return {"encontrado": True, "por_descripcion": False, **d,
                        "cantidad": cantidad, "hay_stock": d["stock"] >= cantidad,
                        "sugerencias": []}
    cands = buscar_productos_por_texto(descripcion or "") if descripcion else []
    # compat: si descripcion era match substring legacy de 1 producto, devolverlo
    if descripcion and not cands:
        p = buscar_producto(descripcion)
        if p:
            d = _producto_a_dict(p)
            return {"encontrado": True, "por_descripcion": True, **d,
                    "cantidad": cantidad, "hay_stock": d["stock"] >= cantidad, "sugerencias": []}
    return {"encontrado": False, "sku_buscado": sku, "descripcion": descripcion,
            "cantidad": cantidad,
            "sugerencias": [{"sku": p["sku"], "nombre": p["nombre"],
                             "precio_lista": float(p["precio_lista"]), "stock": int(p["stock"])}
                            for p in cands],
            "error": f"No encontre '{sku or descripcion}' en el catalogo." if (sku or descripcion) else "Sin criterio"}


@tool(args_schema=VerificarCreditoInput)
def verificar_credito(cuit: str, monto_pedido_neto: Optional[float] = None,
                      monto_neto: Optional[float] = None, pago_anticipado: bool = False) -> dict:
    """Consulta cuenta corriente e informa (art. 7). No confirma ni rechaza."""
    monto = monto_neto if monto_neto is not None else (monto_pedido_neto or 0.0)
    cliente = buscar_cliente(cuit)
    if cliente is None:
        return {"encontrado": False, "cuit": cuit, "entra_en_credito": False,
                "bloqueado": True, "requiere_autorizacion": True,
                "supera_limite": False, "estado": "desconocido", "mora_dias": 0,
                "motivos": [f"CUIT {cuit} no esta en la base"],
                "error": f"No encontre cliente con CUIT '{cuit}'."}
    return reglas.evaluar_credito(cliente, float(monto), bool(pago_anticipado))


@tool(args_schema=CotizarFleteInput)
def cotizar_flete(provincia: str, monto_neto: float, localidad: str = "",
                  pago_anticipado: bool = False, unidades_aerosol: int = 0) -> dict:
    """Cotiza envio por zona: costo, plazo, bonificacion, recargo aerosoles."""
    return reglas.cotizar_flete(provincia, localidad, monto_neto, pago_anticipado, unidades_aerosol)
