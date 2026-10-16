from operator import add
from pydantic import BaseModel, Field, field_validator
from typing import Annotated, Literal, Optional, TypedDict


class EstadoMesa(TypedDict, total=False):
    # Entrada
    mensaje: str
    mensaje_id: str
    cuit: Optional[str]
    cliente: Optional[dict]

    # clasificar (nuevo: dict + intencion; se mantiene string legacy)
    clasificacion: dict | str
    intencion: Literal["consulta", "pedido", "derivar"]
    motivo_clasificacion: str

    # responder_consulta
    respuesta: str
    fuentes: list[dict]
    citas: list[dict]
    ruta: str
    fragmentos: list[dict]

    # extraer_pedido (nuevo: pedido dict; se mantienen items/desc/destino legacy)
    pedido: Optional[dict]
    items: list[dict]
    descuento_solicitado: Optional[float]
    destino: Optional[str]

    # validar_pedido (nuevo: validacion dict; se mantienen pedido/alertas/decision legacy)
    validacion: Optional[dict]
    pedido: dict
    alertas: Annotated[list[str], add]
    decision: Literal["confirmar", "humano", "rechazar", "derivar"]
    motivo_derivacion: str
    motivo_rechazo: str

    # aprobacion_humana (nuevo decision_humana; se mantiene aprobacion legacy)
    decision_humana: Optional[dict]
    aprobacion: dict

    # hojas
    resumen: Optional[dict]
    ticket: Optional[dict]

    # auditoria
    log: Annotated[list[str], add]
    errores: Annotated[list[str], add]


class ClasificacionMensaje(BaseModel):
    """Decision del primer ruteo: que es este mensaje."""
    tipo: Literal["consulta", "pedido", "derivar"] = Field(
        description="consulta = pregunta respondible con los PDF; "
                    "pedido = pide mercaderia con productos y cantidades; derivar = reclamo o tema ajeno"
    )
    intencion: Optional[Literal["consulta", "pedido", "derivar"]] = Field(
        default=None, description="Alias de tipo (compatibilidad con solucion de referencia)"
    )
    confianza: float = Field(default=1.0, ge=0.0, le=1.0, description="Seguridad 0-1")
    motivo: str = Field(description="Por que se clasifico asi, en una frase")
    area_sugerida: Optional[str] = Field(default=None, description="Solo si derivar: area destino")


class ItemPedido(BaseModel):
    sku: Optional[str] = Field(
        default=None,
        description="Codigo XX-0000 si el cliente lo escribio literal. Null si no lo dijo."
    )
    descripcion: str = Field(default="", description="Como nombro el producto el cliente")
    cantidad: int = Field(gt=0, description="Unidades pedidas")

    @field_validator("sku")
    @classmethod
    def normalizar_sku(cls, v):
        if v is None:
            return None
        v = str(v).strip().upper().replace(" ", "")
        return v or None


class PedidoExtraido(BaseModel):
    items: list[ItemPedido]
    descuento_solicitado: Optional[float] = Field(
        default=None, description="Descuento total pedido en % (ej: 20), None si no pidio"
    )
    pago_anticipado: bool = Field(
        default=False, description="True solo si dice que paga antes del envio"
    )
    destino: Optional[str] = Field(
        default=None, description="Localidad/provincia de entrega si la menciona"
    )
    observaciones: str = ""


class RespuestaConsulta(BaseModel):
    respuesta: str
    fuentes: list[int] = Field(default_factory=list, description="Numeros [1..N] de fragmentos usados")
    encontrado: bool = Field(description="False si el contexto no alcanza")


class DecisionHumana(BaseModel):
    tipo: Literal["aprobar", "editar", "rechazar"]
    razon: str = ""
    items_editados: dict[str, int] = Field(default_factory=dict)
    descuento_autorizado: Optional[float] = None
    autorizado_por: str = "Gerencia Comercial"
