"""Config central del proyecto Mesa de Pedidos (paridad con solucion)."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos"
POLITICAS = DATOS / "politicas"
CATALOGO_CSV = DATOS / "catalogo.csv"
CLIENTES_CSV = DATOS / "clientes.csv"
BANDEJA = DATOS / "bandeja.jsonl"
GOLDEN_SET = DATOS / "golden_set.json"

ARTEFACTOS = RAIZ / "artefactos"
INDICE_FAISS = ARTEFACTOS / "indice_faiss"
VECTORSTORE_PATH = ARTEFACTOS / "vectorstore"  # indice legacy local
CHECKPOINTS_DB = ARTEFACTOS / "checkpoints.sqlite"
SALIDAS = ARTEFACTOS / "salidas"

for _c in (ARTEFACTOS, SALIDAS):
    _c.mkdir(parents=True, exist_ok=True)

# RAG
K_RECUPERACION = 4
FETCH_K = 20
LAMBDA_MMR = 0.6
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Reglas de negocio (arts. PDF)
PEDIDO_MINIMO = {"A": 250_000, "B": 250_000, "C": 120_000}
ESCALA_VOLUMEN = [(6_000_000, 12.0), (3_000_000, 8.0), (1_000_000, 5.0), (0, 0.0)]
TOPE_DESCUENTO_GENERAL = 20.0
TOPE_DESCUENTO_MARCAS = 15.0
RUBRO_CON_TOPE_MARCA = "Herramienta electrica"
MARCAS_CON_TOPE = {"dewalt", "makita", "bosch"}
DESCUENTO_PAGO_ANTICIPADO = 3.0
MORA_AUTORIZACION_DIAS = 30
MORA_SUSPENSION_DIAS = 60
CONFIANZA_MINIMA = 0.6

EMPRESA = "Ferreteria Industrial Andina S.R.L."
