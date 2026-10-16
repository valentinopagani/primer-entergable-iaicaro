from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from openai import RateLimitError

from mesa_pedidos.estado import EstadoMesa
from mesa_pedidos.nodos import (
    cargar_cliente, clasificar, responder_consulta, extraer_pedido,
    validar_pedido, aprobacion_humana, confirmar_pedido, rechazar_pedido, derivar_humano,
)


def router_clasificacion(state: EstadoMesa) -> str:
    inten = state.get("intencion")
    if inten in ("consulta", "pedido", "derivar"):
        return inten
    c = state.get("clasificacion")
    if isinstance(c, dict):
        return c.get("intencion") or c.get("tipo") or "derivar"
    return c or "derivar"


def router_validacion(state: EstadoMesa) -> str:
    v = state.get("validacion") or {}
    if v.get("ruta") == "confirmar_pedido":
        return "confirmar"
    if v.get("ruta") == "aprobacion_humana":
        return "humano"
    if v.get("ruta") == "rechazar_pedido":
        return "rechazar"
    if v.get("ruta") == "derivar_humano":
        return "derivar"
    return state.get("decision") or "derivar"


def router_aprobacion(state: EstadoMesa) -> str:
    dh = state.get("decision_humana") or {}
    if dh.get("tipo") in ("aprobar", "editar"):
        return "aprueba"
    if dh.get("tipo") == "rechazar":
        return "rechaza"
    ap = state.get("aprobacion") or {}
    return "aprueba" if ap.get("aprobado") else "rechaza"


def construir_grafo(checkpointer=None):
    grafo = StateGraph(EstadoMesa)

    def _reintentar_sin_cuota(error: Exception) -> bool:
        return not isinstance(error, RateLimitError)

    reintento_llm = RetryPolicy(max_attempts=3, retry_on=_reintentar_sin_cuota)

    grafo.add_node("cargar_cliente", cargar_cliente)
    grafo.add_node("clasificar", clasificar, retry_policy=reintento_llm)
    grafo.add_node("responder_consulta", responder_consulta, retry_policy=reintento_llm)
    grafo.add_node("extraer_pedido", extraer_pedido, retry_policy=reintento_llm)
    grafo.add_node("validar_pedido", validar_pedido)
    grafo.add_node("aprobacion_humana", aprobacion_humana)
    grafo.add_node("confirmar_pedido", confirmar_pedido)
    grafo.add_node("rechazar_pedido", rechazar_pedido)
    grafo.add_node("derivar_humano", derivar_humano)

    grafo.add_edge(START, "cargar_cliente")
    grafo.add_edge("cargar_cliente", "clasificar")
    grafo.add_conditional_edges("clasificar", router_clasificacion,
        {"consulta": "responder_consulta", "pedido": "extraer_pedido", "derivar": "derivar_humano"})
    grafo.add_edge("responder_consulta", END)
    grafo.add_edge("extraer_pedido", "validar_pedido")
    grafo.add_conditional_edges("validar_pedido", router_validacion,
        {"confirmar": "confirmar_pedido", "humano": "aprobacion_humana",
         "rechazar": "rechazar_pedido", "derivar": "derivar_humano"})
    grafo.add_conditional_edges("aprobacion_humana", router_aprobacion,
        {"aprueba": "confirmar_pedido", "rechaza": "rechazar_pedido"})
    grafo.add_edge("confirmar_pedido", END)
    grafo.add_edge("rechazar_pedido", END)
    grafo.add_edge("derivar_humano", END)
    return grafo.compile(checkpointer=checkpointer)


grafo = construir_grafo()
