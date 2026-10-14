"""RAG sobre los PDF de políticas (Clase 4 + híbrido de Clase 5).

Búsqueda densa (MMR) fusionada con BM25 vía Reciprocal Rank Fusion: la densa
captura lo semántico y BM25 el match literal (SKUs, "zapatos", "higiene").
Índice FAISS persistido (ver indexado.py). La respuesta cita documento y
página. Si el contexto no alcanza, no completa: devuelve SIN_CONTEXTO.
"""

from functools import lru_cache
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from mesa_pedidos.indexado import cargar_indice
from mesa_pedidos.llm import crear_llm

SIN_CONTEXTO = "No tengo esa información."

prompt = ChatPromptTemplate.from_template(
    """Respondé la pregunta usando SOLAMENTE el contexto de abajo.
El contexto viene en fragmentos numerados: revisá CADA uno antes de concluir que un dato no está.
Si algún fragmento menciona el tema, el producto o la zona de la pregunta, respondé con ese dato
(regla, límite, monto, plazo) copiando sus números exactos, aunque sea una regla general.
Incluí siempre en la respuesta los montos, umbrales y plazos exactos que figuran en el contexto
(ej: si la bonificación aplica desde $1.500.000, escribí esa cifra).
Sólo si NINGÚN fragmento trata el tema, respondé exactamente: "No tengo esa información.".
Cuando respondas con datos del contexto, al final aclará de qué documento y en qué página los obtuviste.
No inventes. No mezcles con conocimiento previo.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
)

llm = crear_llm()
K = 8


def _denso(pregunta: str, k: int = K) -> list:
    """Búsqueda densa (MMR con foco en similitud)."""
    return cargar_indice().as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 30, "lambda_mult": 0.8},
    ).invoke(pregunta)


@lru_cache(maxsize=1)
def _bm25(k: int = K):
    """Retriever BM25 sobre los mismos chunks del índice (match literal)."""
    from langchain_community.retrievers import BM25Retriever

    chunks = list(cargar_indice().docstore._dict.values())
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever


def _fusion_rrf(listas: list[list], k: int = K, c: int = 60) -> list:
    """Reciprocal Rank Fusion: combina rankings sin calibrar escalas."""
    pesos: dict[str, float] = {}
    por_doc: dict[str, object] = {}
    for lista in listas:
        for rank, doc in enumerate(lista):
            clave = doc.page_content
            pesos[clave] = pesos.get(clave, 0.0) + 1.0 / (c + rank)
            por_doc[clave] = doc
    orden = sorted(pesos, key=lambda c_: pesos[c_], reverse=True)[:k]
    return [por_doc[c_] for c_ in orden]


def recuperar(pregunta: str, k: int = K) -> list:
    """Fragmentos más relevantes: densa + BM25 fusionados (híbrido)."""
    densos = _denso(pregunta, k)
    try:
        lexicales = _bm25(k).invoke(pregunta)
    except Exception:  # si BM25 falla, no rompemos el RAG: densa sola
        return densos
    return _fusion_rrf([densos, lexicales], k)


def format_docs(docs) -> str:
    """Une los fragmentos numerados y etiquetados con documento y página."""
    return "\n\n".join(
        f"[fragmento {i}] [fuente: {Path(d.metadata.get('source', '?')).name} "
        f"p.{d.metadata.get('page', '?')}]\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )


def responder(pregunta: str) -> dict:
    """Contesta una pregunta contra los PDF y devuelve respuesta + fuentes."""
    docs = recuperar(pregunta)
    if not docs:
        return {"respuesta": SIN_CONTEXTO, "fuentes": []}

    contexto = format_docs(docs)
    cadena = (
        {"context": lambda _: contexto, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    respuesta = cadena.invoke(pregunta).strip()
    if respuesta.lower().startswith("no tengo"):
        respuesta = SIN_CONTEXTO

    fuentes = [
        {
            "documento": Path(d.metadata.get("source", "?")).name,
            "pagina": d.metadata.get("page"),
        }
        for d in docs
    ]
    return {"respuesta": respuesta, "fuentes": fuentes}
