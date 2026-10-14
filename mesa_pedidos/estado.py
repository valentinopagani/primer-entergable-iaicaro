from operator import add
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Optional, TypedDict

class EstadoMesa(TypedDict, total=False): # con total=False los nodos devuelven solo los campos que cambian, no el estado entero.
    # Entrada
    mensaje: str

    # cargar_cliente
    cuit: Optional[str]
    cliente: Optional[dict] # fila de clientes.csv + crédito disponible

    # clasificar
    clasificacion: Literal["consulta", "pedido", "derivar"]
    motivo_clasificacion: str

    # responder_consulta
    respuesta: str
    fuentes: list[dict]

    # extraer_pedido
    items: list[dict] # [{"sku": "AB-1020", "cantidad": 200}, ...]
    descuento_solicitado: Optional[float]
    destino: Optional[str]

    # validar_pedido
    pedido: dict # renglones + neto + IVA + total
    alertas: Annotated[list[str], add]
    decision: Literal["confirmar", "humano", "rechazar", "derivar"]
    motivo_derivacion: str
    motivo_rechazo: str

    # aprobacion_humana
    aprobacion: dict

    # auditoría (el reducer hace que NO se pise entre nodos)
    log: Annotated[list[str], add]


class ClasificacionMensaje(BaseModel):
    """Decisión del primer ruteo: qué es este mensaje."""
    tipo: Literal["consulta", "pedido", "derivar"] = Field(
        description="consulta = pregunta respondible con los PDF; "
                    "pedido = pide mercadería; derivar = reclamo o tema que no es ni una ni otra"
    )
    motivo: str = Field(description="Por qué se clasificó así, en una frase")


class ItemPedido(BaseModel):
    sku: str = Field(
        description="SKU del catálogo. Si el cliente nombró un SKU exacto (ej: HE-4330), usá ese literal; "
                    "si no, elegí el SKU que mejor coincide con el nombre pedido."
    )
    cantidad: float = Field(description="Unidades pedidas de ese producto")


class PedidoExtraido(BaseModel):
    items: list[ItemPedido]
    descuento_solicitado: Optional[float] = Field(
        default=None, description="Descuento total pedido en % (ej: 20), None si no pidió ninguno"
    )
    destino: Optional[str] = Field(
        default=None, description="Localidad/provincia de entrega si el mensaje la menciona"
    )
    observaciones: str = ""
