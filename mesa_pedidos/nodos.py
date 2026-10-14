import csv
import re

from langgraph.types import interrupt

from mesa_pedidos.estado import EstadoMesa, ClasificacionMensaje, PedidoExtraido
from mesa_pedidos.llm import crear_llm
from mesa_pedidos.retriever import responder as responder_rag
from mesa_pedidos.tools import (
    CSV_PRODUCTOS,
    buscar_cliente,
    consultar_producto,
    verificar_credito,
)

llm = crear_llm()

clasificador = llm.with_structured_output(ClasificacionMensaje)
extractor = llm.with_structured_output(PedidoExtraido)


PATRON_CUIT = re.compile(r"\b\d{2}-?\d{8}-?\d\b")


def _catalogo_para_prompt() -> str:
    """SKU + nombre de los 40 productos, para que el extractor elija el SKU correcto."""
    with open(CSV_PRODUCTOS, encoding="utf-8-sig") as f:
        return "\n".join(f"{p['sku']}: {p['nombre']}" for p in csv.DictReader(f))


CATALOGO = _catalogo_para_prompt()


def _skus_validos() -> set[str]:
    """SKUs exactos del catálogo (mayúsculas), para validar la extracción."""
    with open(CSV_PRODUCTOS, encoding="utf-8-sig") as f:
        return {p["sku"].strip().upper() for p in csv.DictReader(f)}


SKUS_VALIDOS = _skus_validos()


def _prompt_extraccion(mensaje: str) -> str:
    return (
        "Extraé los items y datos del siguiente pedido de un cliente mayorista. "
        "Elegí el SKU exacto del catálogo que figura al final; si el mensaje nombra un SKU "
        "literal (ej: HE-4330), respetá ese literal. "
        "Devolvé ÚNICAMENTE SKUs que figuren en la lista del catálogo.\n\n"
        f"Mensaje: {mensaje}\n\n"
        f"Catálogo (SKU: nombre):\n{CATALOGO}"
    )


# ---------------------------------------------------------------------------
# 1) cargar_cliente — resuelve el CUIT contra clientes.csv (sin LLM)
# ---------------------------------------------------------------------------
def cargar_cliente(state: EstadoMesa) -> dict:
    cuit = state.get("cuit")
    if not cuit:
        match = PATRON_CUIT.search(state["mensaje"])
        cuit = match.group() if match else None

    cliente = buscar_cliente(cuit) if cuit else None
    return {
        "cuit": cuit,
        "cliente": cliente,
        "log": [f"cliente {'resuelto' if cliente else 'no encontrado'}: {cuit}"],
    }


# ---------------------------------------------------------------------------
# 2) clasificar — LLM + structured output (consulta / pedido / derivar)
# ---------------------------------------------------------------------------
def clasificar(state: EstadoMesa) -> dict:
    prompt = (
        "Sos el clasificador de la Mesa de Pedidos de Ferretería Industrial Andina. "
        "Clasificá el mensaje en uno de estos tres valores:\n"
        "- 'pedido': pide mercadería concreta, con productos y cantidades. Si el mensaje nombra "
        "uno o más productos con cantidades es 'pedido', aunque sea informal "
        "(\"mandame...\", \"te paso pedido\", \"necesito N unidades de...\", \"quiero 20 de...\").\n"
        "- 'consulta': pregunta que se responde con los documentos de la empresa: políticas y "
        "operativa (flete, plazos, devoluciones, garantías, percepción de IIBB, vigencia de "
        "presupuestos) o especificaciones/compatibilidad de un producto (si viene con batería, "
        "qué código es compatible, medidas). Preguntar si se puede devolver un producto o si "
        "aplica la garantía es 'consulta'. Preguntar por límites, condiciones, precios o "
        "disponibilidad (\"cuántos me pueden mandar\", \"cuánto sale\") también es 'consulta' "
        "aunque mencione cantidades: 'pedido' requiere intención de compra.\n"
        "- 'derivar': no define un producto y pide asesoramiento o que lo llamen "
        "(\"no sé qué me conviene\", \"¿me pueden llamar?\"); reclamos de un envío ya hecho "
        "(faltante, error de despacho, nota de crédito); o alguien que no es cliente "
        "(oferta de proveedor).\n\n"
        "Ejemplos:\n"
        "\"¿El atornillador XYZ viene con batería?\" -> consulta\n"
        "\"Compré un producto y me equivoqué de talle, ¿lo puedo devolver?\" -> consulta\n"
        "\"Mandame 20 unidades y 15 cajas a mi sucursal\" -> pedido\n"
        "\"El remito llegó con faltantes, quiero la nota de crédito\" -> derivar\n"
        "\"No sé qué compresor me conviene, ¿me llaman?\" -> derivar\n\n"
        f"Mensaje: {state['mensaje']}"
    )
    r = clasificador.invoke(prompt)  # devuelve un ClasificacionMensaje
    return {
        "clasificacion": r.tipo,
        "motivo_clasificacion": r.motivo,
        "log": [f"clasificación: {r.tipo} — {r.motivo}"],
    }


# ---------------------------------------------------------------------------
# 3) responder_consulta — RAG del 4.1, responde citando documento y página
# ---------------------------------------------------------------------------
def responder_consulta(state: EstadoMesa) -> dict:
    r = responder_rag(state["mensaje"])
    return {
        "respuesta": r["respuesta"],
        "fuentes": r["fuentes"],
        "log": ["consulta respondida con RAG (densa + cita)"],
    }


# ---------------------------------------------------------------------------
# 4) extraer_pedido — LLM + structured output (items, descuento, destino)
# ---------------------------------------------------------------------------
def extraer_pedido(state: EstadoMesa) -> dict:
    prompt = _prompt_extraccion(state["mensaje"])
    r = extractor.invoke(prompt)
    reintento = False

    # Si ningún SKU existe en el catálogo, la extracción falló: un reintento
    # antes de entregar (validación dentro del nodo, sin LLM extra en el camino feliz).
    if r.items and not any(i.sku.strip().upper() in SKUS_VALIDOS for i in r.items):
        reintento = True
        r = extractor.invoke(
            prompt + "\n\nATENCIÓN: los SKU que devolviste no existen en el catálogo. "
            "Volvé a extraer usando ÚNICAMENTE SKUs literales de la lista de arriba."
        )

    log = [f"pedido extraído: {len(r.items)} items"]
    if reintento:
        log.append("extracción reintentada: los SKU del primer intento no existían")
    return {
        "items": [i.model_dump() for i in r.items],  # a dict, para que el estado sea serializable
        "descuento_solicitado": r.descuento_solicitado,
        "destino": r.destino,
        "log": log,
    }


# ---------------------------------------------------------------------------
# 5) validar_pedido — tools + reglas comerciales, SIN LLM
# ---------------------------------------------------------------------------
def validar_pedido(state: EstadoMesa) -> dict:
    cliente = state.get("cliente")
    items = state.get("items") or []

    if cliente is None:
        return {
            "decision": "derivar",
            "motivo_derivacion": "No se pudo identificar al cliente (CUIT ausente o inexistente)",
            "log": ["validación: sin cliente → derivar"],
        }
    if not items:
        return {
            "decision": "derivar",
            "motivo_derivacion": "No se extrajo ningún ítem del mensaje",
            "log": ["validación: pedido sin ítems → derivar"],
        }

    alertas = []
    renglones = []

    for it in items:
        producto = consultar_producto.invoke({"sku_o_nombre": it["sku"]})
        if producto.get("error"):
            alertas.append(f"Producto no encontrado: {it['sku']}")
            continue

        cantidad = float(it["cantidad"])
        if cantidad <= 0:
            alertas.append(f"Cantidad inválida de {producto['sku']}: {cantidad:g}")
            continue
        subtotal = producto["precio_lista"] * cantidad

        if cantidad > producto["stock"]:
            alertas.append(
                f"Stock insuficiente de {producto['sku']}: pedís {cantidad:g}, hay {producto['stock']}"
            )

        renglones.append({
            "sku": producto["sku"],
            "nombre": producto["nombre"],
            "rubro": producto["rubro"],
            "marca": producto["marca"],
            "cantidad": cantidad,
            "precio_unitario": producto["precio_lista"],
            "subtotal": subtotal,
            "deposito": producto["deposito"],
        })

    if not renglones:
        return {
            "decision": "derivar",
            "motivo_derivacion": "Ninguno de los ítems del pedido existe en el catálogo",
            "alertas": alertas,
            "log": ["validación: sin renglones válidos → derivar"],
        }

    neto = sum(r["subtotal"] for r in renglones)

    # Escala de descuento por volumen (Art. 4)
    if neto >= 6_000_000:
        desc_volumen = 12.0
    elif neto >= 3_000_000:
        desc_volumen = 8.0
    elif neto >= 1_000_000:
        desc_volumen = 5.0
    else:
        desc_volumen = 0.0

    desc_habitual = float(cliente["descuento_habitual"])
    desc_solicitado = float(state.get("descuento_solicitado") or 0.0)
    desc_automatico = desc_habitual + desc_volumen

    # Tope de descuento (Art. 5): 15% para herramienta eléctrica Dewalt/Makita/Bosch
    tope = 20.0
    if any(
        r["rubro"] == "Herramienta electrica" and r["marca"] in {"Dewalt", "Makita", "Bosch"}
        for r in renglones
    ):
        tope = 15.0

    # El descuento automático (habitual + volumen) se TOPEA sin consultar (Art. 5).
    # Sólo escala a Gerencia si el cliente PIDE un descuento que hace superar el tope.
    descuento_total = min(desc_automatico + desc_solicitado, tope)
    if desc_solicitado and (desc_automatico + desc_solicitado) > tope:
        alertas.append(
            f"Descuento solicitado de {desc_solicitado:.1f}% lleva el total a "
            f"{desc_automatico + desc_solicitado:.1f}%, por encima del tope de "
            f"{tope:.0f}% — requiere Gerencia"
        )

    # Límite de crédito y mora (Art. 7)
    credito = verificar_credito.invoke({
        "cuit": cliente["cuit"],
        "monto_pedido_neto": neto,
    })
    if credito.get("error"):
        alertas.append(f"Crédito no verificable: {credito['error']}")
    else:
        if credito["supera_limite"]:
            alertas.append(
                f"Supera el límite de crédito: proyectado ${credito['saldo_proyectado']:,.0f} "
                f"vs límite ${credito['limite_credito']:,.0f}"
            )
        if credito["estado"] == "suspendido" or credito["mora_dias"] > 60:
            alertas.append("Cliente suspendido por mora (> 60 días): no se toma a cuenta corriente")
        elif credito["mora_dias"] >= 30:
            alertas.append(f"Cliente en mora ({credito['mora_dias']} días): requiere Gerencia")

    if any("suspendido" in a for a in alertas):
        decision = "rechazar"
    elif alertas:
        decision = "humano"
    else:
        decision = "confirmar"

    iva = neto * 0.21
    pedido = {
        "renglones": renglones,
        "neto": round(neto, 2),
        "descuento_total": descuento_total,
        "tope_descuento": tope,
        "iva": round(iva, 2),
        "total": round(neto + iva, 2),
        "credito": credito,
    }

    return {
        "pedido": pedido,
        "alertas": alertas,
        "decision": decision,
        "log": [
            f"validación: neto ${neto:,.0f}, descuento {descuento_total:.1f}%, "
            f"decisión={decision} ({len(alertas)} alertas)"
        ],
    }


# ---------------------------------------------------------------------------
# 6) aprobacion_humana — frena y pide aprobación a la Gerencia (4.5)
# ---------------------------------------------------------------------------
def aprobacion_humana(state: EstadoMesa) -> dict:
    """Human-in-the-loop: frena el grafo y pide aprobación a Gerencia Comercial.

    Sólo se alcanza cuando validar_pedido decidió "humano" (exceso de crédito,
    descuento sobre el tope, mora, etc.). El payload lleva todo lo que la persona
    necesita para decidir sin abrir otra pantalla: razón social, crédito
    disponible, neto, alertas y renglones.

    La ejecución se suspende en interrupt(); el proceso termina y el estado
    queda en el checkpointer. Se retoma en otra corrida con:
        Command(resume={"aprobado": bool, "motivo": str})
    y recién ahí interrupt() devuelve ese valor y el nodo completa.
    """
    cliente = state["cliente"]
    pedido = state["pedido"]
    credito = pedido.get("credito", {})

    payload = {
        "accion": "aprobar_pedido",
        "razon_social": cliente["razon_social"],
        "cuit": cliente["cuit"],
        "categoria": cliente["categoria"],
        "credito_disponible": credito.get("credito_disponible"),
        "limite_credito": credito.get("limite_credito"),
        "saldo_actual": credito.get("saldo_actual"),
        "neto": pedido["neto"],
        "descuento_total": pedido["descuento_total"],
        "iva": pedido["iva"],
        "total": pedido["total"],
        "alertas": list(state.get("alertas", [])),
        "renglones": pedido["renglones"],
        "options": ["aprobar", "rechazar"],
    }

    decision = interrupt(payload)
    decision = decision or {}
    motivo = decision.get("motivo", "")

    if decision.get("aprobado"):
        return {
            "aprobacion": {"aprobado": True, "motivo": motivo},
            "log": ["aprobación humana: APROBADO por Gerencia Comercial"],
        }

    return {
        "aprobacion": {"aprobado": False, "motivo": motivo},
        "motivo_rechazo": motivo or "rechazado por Gerencia Comercial",
        "log": [f"aprobación humana: RECHAZADO — {motivo or 'sin motivo'}"],
    }


# ---------------------------------------------------------------------------
# 7) confirmar_pedido
# ---------------------------------------------------------------------------
def confirmar_pedido(state: EstadoMesa) -> dict:
    pedido = state["pedido"]
    cliente = state["cliente"]
    return {
        "respuesta": (
            f"Pedido confirmado para {cliente['razon_social']}. "
            f"Neto ${pedido['neto']:,.2f} + IVA = total ${pedido['total']:,.2f}."
        ),
        "log": ["pedido confirmado"],
    }


# ---------------------------------------------------------------------------
# 8) rechazar_pedido
# ---------------------------------------------------------------------------
def rechazar_pedido(state: EstadoMesa) -> dict:
    motivo = state.get("motivo_rechazo") or "; ".join(state.get("alertas", [])) or "rechazado"
    return {
        "respuesta": f"Pedido rechazado: {motivo}.",
        "log": [f"pedido rechazado — {motivo}"],
    }


# ---------------------------------------------------------------------------
# 9) derivar_humano — lo alcanzan la clasificación 'derivar' y la validación 'derivar'
# ---------------------------------------------------------------------------
def derivar_humano(state: EstadoMesa) -> dict:
    # El motivo que ve el usuario es determinista; el texto del clasificador
    # sólo queda en el log de auditoría.
    if state.get("motivo_derivacion"):
        motivo_usuario = state["motivo_derivacion"]
    elif state.get("clasificacion") == "derivar":
        motivo_usuario = (
            "es un reclamo o un pedido de asesoramiento que no es una consulta "
            "de política ni un pedido nuevo"
        )
    else:
        motivo_usuario = "requiere intervención humana"

    motivo_log = state.get("motivo_clasificacion") or motivo_usuario
    return {
        "respuesta": f"Derivado a una persona: {motivo_usuario}.",
        "log": [f"derivado a humano — {motivo_log}"],
    }
