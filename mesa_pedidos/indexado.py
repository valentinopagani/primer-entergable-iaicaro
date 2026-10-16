"""Indexado de los PDF de políticas (punto 4.1).

Carga los 4 PDF de datos/politicas, los parte en chunks, genera los
embeddings con un modelo denso local (sentence-transformers/all-MiniLM-L6-v2,
sin cuota de API) y persiste el índice FAISS en artefactos/vectorstore.

Reindexar en cada corrida es tirar plata: si el índice ya existe,
cargar_indice() lo levanta del disco y no vuelve a construirlo.

Se corre una vez con:
    python mesa_pedidos/indexado.py
"""

import warnings
from functools import lru_cache
from pathlib import Path

# NOTA: los imports pesados (langchain_community, torch, transformers) estan
# DENTRO de las funciones a proposito: importarlos al inicio dejaba ~1 min
# sin ningun output y parecia que el programa estaba clavado.
warnings.filterwarnings("ignore", message=".*HuggingFaceEmbeddings.*")

BASE_DIR = Path(__file__).resolve().parent.parent  # .../Mesa de Pedidos
DIR_PDF = BASE_DIR / "datos" / "politicas"
VECTORSTORE_PATH = BASE_DIR / "artefactos" / "vectorstore"

EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def crear_embeddings():
    """Devuelve el modelo de embeddings denso local (sin API ni cuota)."""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)


def cargar_documentos() -> list:
    """Carga los 4 PDF con PyPDFLoader.

    PyPDFLoader numerá las páginas desde 0; acá se deja de 1 a N para que
    la cita que ve una persona coincida con lo que muestra el PDF.
    """
    from langchain_community.document_loaders import PyPDFLoader

    pdfs = sorted(DIR_PDF.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No se encontraron PDF en {DIR_PDF}")

    documentos = []
    for pdf in pdfs:
        docs = PyPDFLoader(str(pdf)).load()
        for doc in docs:
            doc.metadata["page"] = int(doc.metadata.get("page", 0)) + 1
        documentos.extend(docs)
        print(f"---- {pdf.name}: {len(docs)} páginas cargadas ----")
    return documentos


def construir_indice():
    """Parte los PDF en chunks, genera los embeddings y persiste el índice."""
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    documentos = cargar_documentos()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documentos)
    print(f"---- {len(chunks)} chunks generados ----")

    embeddings = crear_embeddings()
    print("---- Embeddings generados ----")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_PATH))
    print(f"---- Índice persistido en {VECTORSTORE_PATH} ----")
    return vectorstore


@lru_cache(maxsize=1)
def cargar_indice() -> FAISS:
    """Levanta el índice persistido; si no existe, lo construye una vez.

    Cacheada con lru_cache: el modelo de embeddings y el vectorstore se cargan
    una sola vez por proceso, aunque se consulte muchas veces.
    """
    from langchain_community.vectorstores import FAISS

    if not (VECTORSTORE_PATH / "index.faiss").exists():
        return construir_indice()

    embeddings = crear_embeddings()
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"---- Índice cargado desde {VECTORSTORE_PATH} ----")
    return vectorstore


if __name__ == "__main__":
    construir_indice()
