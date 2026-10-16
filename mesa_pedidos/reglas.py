"""Reglas comerciales puras (arts. PDF). Sin LLM, sin archivos, sin grafo."""
from __future__ import annotations
import re
import unicodedata
from typing import Any, Optional

import config


def normalizar_texto(t: str) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFD", t.lower()).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", t).strip()


def pesos(monto: float) -> str:
    s = f"{monto:,.2f}"
    return "$" + s.replace(",", "_").replace(".", ",").replace("_", ".")


def porcentaje(valor: float) -> str:
    texto = f"{valor:.1f}".replace(".", ",")
    return (texto[:-2] if texto.endswith(",0") else texto) + "%"


# --- intencion efectiva ---
def intencion_efectiva(clasificacion: dict, cliente: Optional[dict]) -> tuple[str, str]:
    intencion = clasificacion.get("intencion") or clasificacion.get("tipo", "derivar")
    try:
        confianza = float(clasificacion.get("confianza", 1.0))
    except Exception:
        confianza = 1.0
    if confianza < config.CONFIANZA_MINIMA:
        return "derivar", f"confianza {confianza:.2f} < {config.CONFIANZA_MINIMA}: se deriva"
    if intencion == "pedido" and not cliente:
        return "derivar", "pedido de un CUIT que no esta en la base: se deriva"
    if intencion not in ("consulta", "pedido", "derivar"):
        return "derivar", f"intencion desconocida '{intencion}': se deriva"
    return intencion, ""


# --- descuentos / valorizacion ---
def descuento_por_volumen(monto_neto: float) -> float:
    for desde, pct in config.ESCALA_VOLUMEN:
        if monto_neto >= desde:
            return pct
    return 0.0


def tope_del_producto(rubro: str, marca: str) -> float:
    if (normalizar_texto(rubro) == normalizar_texto(config.RUBRO_CON_TOPE_MARCA)
            and normalizar_texto(marca) in config.MARCAS_CON_TOPE):
        return config.TOPE_DESCUENTO_MARCAS
    return config.TOPE_DESCUENTO_GENERAL


def valorizar(items: list[dict], cliente: dict,
              descuento_solicitado: Optional[float] = None,
              pago_anticipado: bool = False,
              descuento_autorizado: Optional[float] = None) -> dict[str, Any]:
    bruto = sum(i["cantidad"] * i["precio_lista"] for i in items)
    volumen = descuento_por_volumen(bruto)
    comercial = float(cliente.get("descuento_habitual", 0.0)) + volumen
    notas = [f"Descuento habitual {porcentaje(float(cliente.get('descuento_habitual', 0)))} "
             f"+ volumen {porcentaje(volumen)} (art. 4) = {porcentaje(comercial)}"]
    if descuento_autorizado is not None:
        general = float(descuento_autorizado)
        notas.append(f"Descuento autorizado por Gerencia: {porcentaje(general)}")
    else:
        if descuento_solicitado and comercial < descuento_solicitado <= config.TOPE_DESCUENTO_GENERAL:
            comercial = float(descuento_solicitado)
            notas.append(f"Bonificacion puntual pedida por el cliente: {porcentaje(comercial)} (art. 5)")
        general = min(comercial, config.TOPE_DESCUENTO_GENERAL)
        if general < comercial:
            notas.append(f"Topeado al {porcentaje(general)} (art. 5)")

    renglones = []
    for i in items:
        tope = tope_del_producto(i.get("rubro", ""), i.get("marca", ""))
        pct = min(general, tope)
        subtotal_lista = i["cantidad"] * i["precio_lista"]
        renglones.append({
            "sku": i["sku"], "nombre": i["nombre"], "marca": i.get("marca", ""),
            "rubro": i.get("rubro", ""), "cantidad": i["cantidad"],
            "precio_unitario": i["precio_lista"],
            "subtotal_lista": round(subtotal_lista, 2),
            "descuento_pct": pct,
            "subtotal_neto": round(subtotal_lista * (1 - pct / 100), 2),
            "subtotal": round(subtotal_lista * (1 - pct / 100), 2),
            "alicuota_iva": i.get("alicuota_iva", 21.0),
            "stock_disponible": i.get("stock", 0), "stock": i.get("stock", 0),
            "hay_stock": i.get("stock", 0) >= i["cantidad"],
            "deposito": i.get("deposito", ""),
            "nota": f"tope {porcentaje(tope)} por marca (art. 5)" if pct < general else "",
        })
    neto_merc = sum(r["subtotal_neto"] for r in renglones)
    factor_fin = config.DESCUENTO_PAGO_ANTICIPADO / 100 if pago_anticipado else 0.0
    desc_fin = round(neto_merc * factor_fin, 2)
    if pago_anticipado:
        notas.append(f"Pago anticipado: {porcentaje(config.DESCUENTO_PAGO_ANTICIPADO)} financiero, fuera del tope (art. 6)")
    neto = round(neto_merc - desc_fin, 2)
    iva = round(sum(r["subtotal_neto"] * r["alicuota_iva"] / 100 for r in renglones) * (1 - factor_fin), 2)
    return {"renglones": renglones, "bruto": round(bruto, 2),
            "descuento_volumen_pct": volumen, "descuento_aplicado_pct": general,
            "descuento_financiero": desc_fin, "neto": neto, "iva": iva, "notas": notas}


# --- credito art.7 ---
def evaluar_credito(cliente: dict, monto_neto: float, pago_anticipado: bool = False) -> dict[str, Any]:
    limite = float(cliente["limite_credito"])
    saldo = float(cliente["saldo_actual"])
    mora = int(cliente["mora_dias"])
    suspendido = cliente["estado"] == "suspendido" or mora > config.MORA_SUSPENSION_DIAS
    mora_aut = config.MORA_AUTORIZACION_DIAS <= mora <= config.MORA_SUSPENSION_DIAS
    proyectado = saldo if pago_anticipado else saldo + monto_neto
    excede = proyectado > limite
    motivos = []
    if suspendido:
        motivos.append(f"cliente {cliente['estado']} con {mora} dias de mora (art. 7)")
    if mora_aut:
        motivos.append(f"mora de {mora} dias: entre {config.MORA_AUTORIZACION_DIAS} y "
                       f"{config.MORA_SUSPENSION_DIAS} requiere autorizacion (art. 7)")
    if excede:
        motivos.append(f"saldo {pesos(saldo)} + pedido {pesos(monto_neto)} = "
                       f"{pesos(proyectado)} supera el limite de {pesos(limite)} (art. 7)")
    disponible = max(0.0, limite - saldo)
    return {"encontrado": True, "razon_social": cliente.get("razon_social"),
            "cuit": cliente.get("cuit"), "categoria": cliente.get("categoria"),
            "provincia": cliente.get("provincia"), "estado": cliente["estado"], "mora_dias": mora,
            "limite_credito": limite, "saldo_actual": saldo, "credito_disponible": disponible,
            "monto_neto": monto_neto, "monto_pedido_neto": monto_neto,
            "saldo_proyectado": proyectado, "pago_anticipado": pago_anticipado,
            "supera_limite": excede,
            "entra_en_credito": pago_anticipado or (not suspendido and not excede),
            "bloqueado": suspendido and not pago_anticipado,
            "requiere_autorizacion": (excede and not suspendido) or mora_aut,
            "motivos": motivos,
            "descuento_habitual": float(cliente.get("descuento_habitual", 0.0))}


# --- flete ---
ZONAS_POR_PROVINCIA = {
    2: {"cordoba", "santa fe", "entre rios"},
    3: {"buenos aires", "la pampa", "san luis", "mendoza", "san juan"},
    4: {"salta", "jujuy", "tucuman", "santiago del estero", "catamarca", "la rioja",
        "chaco", "corrientes", "formosa", "misiones"},
    5: {"neuquen", "rio negro", "chubut", "santa cruz", "tierra del fuego"},
}
GRAN_CORDOBA = {"cordoba", "cordoba capital", "villa allende", "mendiolaza", "saldan",
                "la calera", "malvinas argentinas", "rio ceballos", "unquillo", "malagueno"}
TARIFAS = {1: (35_000, "24 a 48 horas habiles", "Cruz del Sur"),
           2: (68_000, "3 a 4 dias habiles", "Cruz del Sur"),
           3: (95_000, "4 a 6 dias habiles", "Cruz del Sur"),
           4: (130_000, "6 a 8 dias habiles", "Andesmar Cargas"),
           5: (175_000, "8 a 12 dias habiles", "Andesmar Cargas")}
UMBRAL_BONIFICACION = 1_500_000
UMBRAL_BONIFICACION_TOTAL_ZONAS_4_5 = 3_000_000
MAX_AEROSOLES_ZONAS_4_5 = 24
RECARGO_PELIGROSA = 0.15


def zona_de(provincia: str, localidad: str = "") -> Optional[int]:
    prov = normalizar_texto(provincia).strip()
    if prov == "cordoba" and normalizar_texto(localidad).strip() in GRAN_CORDOBA:
        return 1
    for zona, provs in ZONAS_POR_PROVINCIA.items():
        if prov in provs:
            return zona
    return None


def cotizar_flete(provincia: str, localidad: str, monto_neto: float,
                  pago_anticipado: bool = False, unidades_aerosol: int = 0) -> dict[str, Any]:
    zona = zona_de(provincia, localidad)
    if zona is None:
        return {"zona": None, "costo": 0.0, "plazo": "a convenir", "transportista": "",
                "bonificacion_pct": 0, "alertas": [],
                "detalle": f"Provincia '{provincia}' sin zona asignada"}
    base, plazo, transp = TARIFAS[zona]
    if zona <= 3:
        bonif = 100 if monto_neto > UMBRAL_BONIFICACION else 0
    else:
        bonif = (100 if monto_neto >= UMBRAL_BONIFICACION_TOTAL_ZONAS_4_5
                 else 50 if monto_neto >= UMBRAL_BONIFICACION else 0)
    detalle = [f"Zona {zona} ({provincia}): {pesos(base)}, {plazo}, {transp}"]
    if bonif and pago_anticipado:
        detalle.append("La bonificacion de flete no se acumula con el descuento por pago anticipado (punto 3)")
        bonif = 0
    elif bonif:
        detalle.append(f"Flete bonificado al {bonif}% por monto (punto 3)")
    costo = base * (1 - bonif / 100)
    alertas: list[str] = []
    if unidades_aerosol:
        costo *= 1 + RECARGO_PELIGROSA
        detalle.append("Recargo 15% por carga peligrosa (aerosoles, punto 5)")
        if zona >= 4 and unidades_aerosol > MAX_AEROSOLES_ZONAS_4_5:
            alertas.append(f"{unidades_aerosol} aerosoles a zona {zona}: maximo "
                           f"{MAX_AEROSOLES_ZONAS_4_5} por envio (Logistica, punto 5)")
    return {"zona": zona, "costo": round(costo, 2), "costo_base": base, "plazo": plazo,
            "transportista": transp, "bonificacion_pct": bonif,
            "alertas": alertas, "detalle": "; ".join(detalle)}


# --- montos en consultas ---
_RE_MONTO = re.compile(r"\$?\s?(\d{1,3}(?:\.\d{3})+|\d{4,})(?:,\d+)?")


def montos_en_texto(texto: str) -> list[float]:
    montos = []
    for m in _RE_MONTO.finditer(texto or ""):
        valor = float(m.group(1).replace(".", ""))
        if valor >= 10_000:
            montos.append(valor)
    return montos


def calculos_para_consulta(texto: str, cliente: Optional[dict]) -> str:
    lineas = []
    for monto in montos_en_texto(texto):
        lineas.append(f"Para un pedido de {pesos(monto)} neto: descuento por volumen "
                      f"{porcentaje(descuento_por_volumen(monto))} (art. 4)")
        if cliente:
            flete = cotizar_flete(cliente.get("provincia", ""), cliente.get("localidad", ""), monto)
            if flete["zona"] is None:
                continue
            lineas.append(f"  Flete a {cliente.get('localidad')} ({cliente.get('provincia')}): {flete['detalle']}. "
                          f"Con {pesos(monto)} el flete cuesta {pesos(flete['costo'])}.")
    return "\n".join(lineas) or "(la pregunta no menciona montos)"


# --- decision ---
def decidir_ruta(cliente: dict, valorizacion: dict, credito: dict, flete: dict,
                no_resueltos: list[dict], inferidos: list[str],
                descuento_solicitado: Optional[float] = None) -> tuple[str, str, list[str]]:
    renglones = valorizacion.get("renglones", [])
    if not renglones:
        return ("derivar_humano",
                "No se pudo identificar ningun producto del catalogo en el pedido",
                [f"Item no identificado: '{i.get('descripcion')}'" for i in no_resueltos])
    if credito.get("bloqueado"):
        return ("rechazar_pedido",
                f"Cliente {cliente['estado']} con {cliente['mora_dias']} dias de mora: "
                "no se toman pedidos en cuenta corriente, solo con pago anticipado (art. 7)",
                list(credito.get("motivos", [])))
    alertas: list[str] = []
    if credito.get("requiere_autorizacion"):
        alertas += credito.get("motivos", [])
    if cliente.get("categoria") == "C" and not credito.get("pago_anticipado"):
        alertas.append("Cliente categoria C sin pago anticipado: la cuenta corriente "
                       "requiere autorizacion de Gerencia (art. 6)")
    if descuento_solicitado and descuento_solicitado > config.TOPE_DESCUENTO_GENERAL:
        alertas.append(f"Pide {porcentaje(descuento_solicitado)} de descuento: supera el tope de "
                       f"{porcentaje(config.TOPE_DESCUENTO_GENERAL)} (art. 5)")
    for r in renglones:
        if not r.get("hay_stock", True):
            alertas.append(f"Sin stock suficiente de {r['sku']}: piden {r['cantidad']}, hay "
                           f"{r.get('stock_disponible', r.get('stock', 0))} (art. 9)")
    for sku in inferidos:
        alertas.append(f"{sku} se identifico por la descripcion, no por el codigo: confirmar")
    for i in no_resueltos:
        alertas.append(f"Item no identificado en el catalogo: '{i.get('descripcion')}'")
    minimo = config.PEDIDO_MINIMO.get(cliente.get("categoria", "B"), 0)
    if valorizacion.get("neto", 0) < minimo:
        alertas.append(f"Neto {pesos(valorizacion.get('neto', 0))} por debajo del minimo de "
                       f"{pesos(minimo)} para categoria {cliente.get('categoria')} (art. 3)")
    alertas += flete.get("alertas", [])
    if alertas:
        return "aprobacion_humana", alertas[0], alertas
    return ("confirmar_pedido",
            "Dentro del credito, con stock y con el descuento dentro del tope", [])
