"""Configuración compartida del LLM (OpenRouter)."""

import os

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv(usecwd=True))

MODELO = os.getenv("MODELO_LLM", "openai/gpt-oss-20b")
MAX_TOKENS = 4096


def crear_llm(max_tokens: int = MAX_TOKENS) -> ChatOpenAI:
    """Devuelve un ChatOpenAI contra OpenRouter con temperatura 0.

    Sin reintentos internos del cliente (max_retries=0): el reintento lo maneja
    el RetryPolicy del grafo, para no apilar 2 (cliente) x 3 (grafo) esperas
    ante un proveedor lento.
    """
    return ChatOpenAI(
        model=MODELO,
        temperature=0,
        max_tokens=max_tokens,
        request_timeout=60,
        max_retries=0,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )
