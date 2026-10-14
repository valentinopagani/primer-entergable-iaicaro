from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from openai import RateLimitError

from mesa_pedidos.estado import EstadoMesa
from mesa_pedidos.nodos import (
    cargar_cliente,
    clasificar,
    responder_consulta,
    extraer_pedido,
    validar_pedido,
    aprobacion_humana,
    confirmar_pedido,
    rechazar_pedido,
    derivar_humano,
)


# ---------------------------------------------------------------------------
# Routers: leen el estado y devuelven un string. NO toman decisiones de negocio,
# solo traducen una decisión que ya escribió un nodo.
# ---------------------------------------------------------------------------
def router_clasificacion(state: EstadoMesa) -> str:
    return state["clasificacion"]          # "consulta" | "pedido" | "derivar"


def router_validacion(state: EstadoMesa) -> str:
    return state["decision"]               # "confirmar" | "humano" | "rechazar" | "derivar"


def router_aprobacion(state: EstadoMesa) -> str:
    # El nodo aprobacion_humana escribió "aprobacion" al volver de interrupt().
    aprobacion = state.get("aprobacion") or {}
    return "aprueba" if aprobacion.get("aprobado") else "rechaza"


def construir_grafo(checkpointer=None):
    grafo = StateGraph(EstadoMesa)

    # RetryPolicy (Clase 7): los nodos con LLM reintentan ante fallos
    # intermitentes (p. ej. LengthFinishReasonError en structured output),
    # pero NO ante errores de cuota (429): reintentar de inmediato sólo
    # suma ráfaga contra un límite ya saturado.
    def _reintentar_sin_cuota(error: Exception) -> bool:
        return not isinstance(error, RateLimitError)

    reintento_llm = RetryPolicy(max_attempts=3, retry_on=_reintentar_sin_cuota)

    # --- los 9 nodos ---
    grafo.add_node("cargar_cliente", cargar_cliente)
    grafo.add_node("clasificar", clasificar, retry_policy=reintento_llm)
    grafo.add_node("responder_consulta", responder_consulta, retry_policy=reintento_llm)
    grafo.add_node("extraer_pedido", extraer_pedido, retry_policy=reintento_llm)
    grafo.add_node("validar_pedido", validar_pedido)
    grafo.add_node("aprobacion_humana", aprobacion_humana)
    grafo.add_node("confirmar_pedido", confirmar_pedido)
    grafo.add_node("rechazar_pedido", rechazar_pedido)
    grafo.add_node("derivar_humano", derivar_humano)

    # --- arranque ---
    grafo.add_edge(START, "cargar_cliente")
    grafo.add_edge("cargar_cliente", "clasificar")

    # --- ruteo 1: clasificación ---
    grafo.add_conditional_edges(
        "clasificar",
        router_clasificacion,
        {
            "consulta": "responder_consulta",
            "pedido": "extraer_pedido",
            "derivar": "derivar_humano",
        },
    )

    grafo.add_edge("responder_consulta", END)   # hoja del RAG

    # --- rama pedido ---
    grafo.add_edge("extraer_pedido", "validar_pedido")

    # --- ruteo 2: validación (el crítico) ---
    grafo.add_conditional_edges(
        "validar_pedido",
        router_validacion,
        {
            "confirmar": "confirmar_pedido",
            "humano": "aprobacion_humana",
            "rechazar": "rechazar_pedido",
            "derivar": "derivar_humano",
        },
    )

    # --- ruteo 3: aprobación humana (4.5) ---
    grafo.add_conditional_edges(
        "aprobacion_humana",
        router_aprobacion,
        {
            "aprueba": "confirmar_pedido",
            "rechaza": "rechazar_pedido",
        },
    )

    # --- hojas ---
    grafo.add_edge("confirmar_pedido", END)
    grafo.add_edge("rechazar_pedido", END)
    grafo.add_edge("derivar_humano", END)

    return grafo.compile(checkpointer=checkpointer)


# instancia lista para importar y dibujar (sin checkpointer todavía: eso es 4.5)
grafo = construir_grafo()
