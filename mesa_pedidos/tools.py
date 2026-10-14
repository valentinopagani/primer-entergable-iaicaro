import csv
from pathlib import Path
from langchain.tools import tool
from pydantic import BaseModel, Field

CSV_PRODUCTOS = Path(__file__).resolve().parent.parent / "datos" / "catalogo.csv"
CSV_CLIENTES = Path(__file__).resolve().parent.parent / "datos" / "clientes.csv"

def _leer_csv(ruta: Path) -> list[dict]:
    """Lee un CSV y devuelve una lista de diccionarios."""
    with open(ruta, encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)

def buscar_producto(consulta: str) -> dict | None:
    """SKU exacto o coincidencia (parcial, sin distinguir mayúsculas) en el nombre."""
    consulta = consulta.strip().upper()
    productos = _leer_csv(CSV_PRODUCTOS)

    for p in productos:
        if (p["sku"].upper() == consulta) or (consulta in p["nombre"].upper()):
            return p
    return None

def buscar_cliente(cuit: str) -> dict | None:
    cuit = cuit.replace("-", "").strip()

    for c in _leer_csv(CSV_CLIENTES):
        if c["cuit"].replace("-", "").strip() == cuit:
            return c
    return None


class ConsultarProductoInput(BaseModel):
    sku_o_nombre: str = Field(
        ...,
        description="SKU exacto del catálogo (ej: 'HE-4330') o nombre del producto (ej: 'disco de corte')",
    )

class VerificarCreditoInput(BaseModel):
    cuit: str = Field(
        ...,
        description="CUIT exacto del cliente (ej: 30-71455982-2)"
    )
    monto_pedido_neto: float = Field(
        ...,
        description="Monto neto del pedido en pesos, sin IVA (ej: 1500000)",
    )

@tool(args_schema=ConsultarProductoInput)
def consultar_producto(sku_o_nombre: str) -> dict:
    """Busca un producto en el catálogo real de 40 productos.

    Devuelve SKU, nombre, rubro, marca, unidad, precio de lista (sin IVA),
    alícuota de IVA, stock disponible y depósito. Si no existe, devuelve
    {'error': ...}.
    """
    producto = buscar_producto(sku_o_nombre)
    if producto is None:
        return {"error": f"No encontré '{sku_o_nombre}' en el catálogo."}

    return {
        "sku": producto["sku"],
        "nombre": producto["nombre"],
        "rubro": producto["rubro"],
        "marca": producto["marca"],
        "unidad": producto["unidad"],
        "precio_lista": float(producto["precio_lista"]),
        "alicuota_iva": float(producto["alicuota_iva"]),
        "stock": int(producto["stock"]),
        "deposito": producto["deposito"],
    }

@tool(args_schema=VerificarCreditoInput)
def verificar_credito(cuit: str, monto_pedido_neto: float) -> dict:
    """Verifica el crédito de un cliente contra su cuenta corriente real.

    Devuelve límite de crédito, saldo actual, crédito disponible, si el pedido
    supera el límite, estado del cliente (activo/suspendido), días de mora y
    descuento habitual. Úsala antes de confirmar cualquier pedido a cuenta corriente.
    """
    cliente = buscar_cliente(cuit)
    if cliente is None:
        return {"error": f"No encontré ningún cliente con CUIT '{cuit}'."}

    limite = float(cliente["limite_credito"])
    saldo = float(cliente["saldo_actual"])
    disponible = limite - saldo
    proyectado = saldo + monto_pedido_neto

    return {
        "cuit": cliente["cuit"],
        "razon_social": cliente["razon_social"],
        "categoria": cliente["categoria"],
        "provincia": cliente["provincia"],
        "limite_credito": limite,
        "saldo_actual": saldo,
        "credito_disponible": disponible,
        "monto_pedido_neto": monto_pedido_neto,
        "saldo_proyectado": proyectado,
        "supera_limite": proyectado > limite,
        "estado": cliente["estado"],
        "mora_dias": int(cliente["mora_dias"]),
        "descuento_habitual": float(cliente["descuento_habitual"]),
        "dias_plazo": int(cliente["dias_plazo"]),
    }
