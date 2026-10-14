"""evaluar.py — corre los 20 mensajes contra el golden set y saca dos números (4.6).

Métricas:
  1. RAMA CORRECTA: cuántos de los 20 casos terminaron en la ruta esperada.
  2. DATO CLAVE: en cuántas de las 8 consultas la respuesta contiene el dato del golden set.

Cada caso corre secuencial en su propio thread_id, con InMemorySaver porque
los casos que escalan a humano disparan interrupt() y necesitan checkpointer.

Uso:
    python evaluar.py
"""

import json
import re
import time
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from mesa_pedidos.grafo import construir_grafo

BASE = Path(__file__).resolve().parent
BANDEJA = BASE / "datos" / "bandeja.jsonl"
GOLDEN = BASE / "datos" / "golden_set.json"
SALIDA = BASE / "artefactos" / "evaluacion.json"

# Pausa entre casos para no saturar el RPM del tier gratuito de Gemini
# (~2 llamadas LLM por caso; la cuota free es de un dígito por minuto).
PAUSA_ENTRE_CASOS = 10


def cargar_jsonl(ruta: Path) -> list[dict]:
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def normalizar(texto: str) -> str:
    """Saca puntos, comas, espacios y guiones para comparar montos y SKUs."""
    return re.sub(r"[.,\s\u2010-\u2015\u2212\-]", "", texto.lower())


def es_no_respuesta(texto: str) -> bool:
    """El sistema no completó: respondió el centinela del 4.1 y no cuenta como acierto."""
    return texto.strip().lower().startswith("no tengo")


def ruta_obtenida(result: dict) -> str:
    """Traduce el estado final del grafo a la ruta (nodo hoja) donde terminó."""
    if "__interrupt__" in result:
        return "aprobacion_humana"

    clasificacion = result.get("clasificacion")
    if clasificacion == "consulta":
        return "responder_consulta"
    if clasificacion == "derivar":
        return "derivar_humano"

    return {
        "confirmar": "confirmar_pedido",
        "rechazar": "rechazar_pedido",
        "derivar": "derivar_humano",
        "humano": "aprobacion_humana",
    }.get(result.get("decision"), "desconocida")


def evaluar_caso(caso: dict, msg: dict) -> dict:
    """Corre un caso de punta a punta y devuelve la fila del reporte."""
    grafo = construir_grafo(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": f"eval-{caso['id']}"}}
    try:
        out = grafo.invoke({"mensaje": msg["texto"], "cuit": msg["cuit"]}, config=config)
        obtenida = ruta_obtenida(out)
        respuesta = out.get("respuesta", "")
    except Exception as e:  # un caso roto no debe tumbar la evaluación
        obtenida = f"ERROR: {type(e).__name__}"
        respuesta = ""

    rama_ok = obtenida == caso["ruta"]

    datos_ok = None
    faltantes: list[str] = []
    if caso["intencion"] == "consulta":
        if es_no_respuesta(respuesta):
            faltantes = list(caso["debe_contener"])
        else:
            faltantes = [
                d for d in caso["debe_contener"] if normalizar(d) not in normalizar(respuesta)
            ]
        datos_ok = not faltantes

    return {
        "id": caso["id"],
        "intencion": caso["intencion"],
        "ruta_esperada": caso["ruta"],
        "ruta_obtenida": obtenida,
        "rama_ok": rama_ok,
        "datos_ok": datos_ok,
        "faltantes": faltantes,
        "respuesta": respuesta,
    }


def main() -> None:
    bandeja = {m["id"]: m for m in cargar_jsonl(BANDEJA)}
    casos = json.loads(GOLDEN.read_text(encoding="utf-8"))["casos"]
    filas = []
    for i, caso in enumerate(casos):
        if i > 0:
            time.sleep(PAUSA_ENTRE_CASOS)
        print(f"... {caso['id']} ({i + 1}/{len(casos)})", flush=True)
        filas.append(evaluar_caso(caso, bandeja[caso["id"]]))

    print("=" * 78)
    print("DETALLE POR CASO")
    print("=" * 78)
    print(f"{'ID':<9} {'intención':<10} {'esperada':<20} {'obtenida':<20} {'rama':<5} {'dato'}")
    print("-" * 78)
    for f in filas:
        dato = "-" if f["datos_ok"] is None else ("ok" if f["datos_ok"] else "FALTA " + ",".join(f["faltantes"]))
        print(
            f"{f['id']:<9} {f['intencion']:<10} {f['ruta_esperada']:<20} "
            f"{f['ruta_obtenida']:<20} {'OK' if f['rama_ok'] else 'NO':<5} {dato}"
        )

    total = len(filas)
    ramas_ok = sum(f["rama_ok"] for f in filas)
    consultas = [f for f in filas if f["intencion"] == "consulta"]
    datos_ok = sum(1 for f in consultas if f["datos_ok"])

    print("\n" + "=" * 78)
    print("RESULTADOS")
    print("=" * 78)
    print(f"1) Rama correcta:            {ramas_ok}/{total}  ({100 * ramas_ok / total:.0f}%)")
    print(f"2) Dato clave en consultas:  {datos_ok}/{len(consultas)}  ({100 * datos_ok / len(consultas):.0f}%)")

    if any(not f["rama_ok"] for f in filas):
        print("\nCasos con rama equivocada:")
        for f in filas:
            if not f["rama_ok"]:
                print(f"  {f['id']}: esperaba {f['ruta_esperada']}, obtuvo {f['ruta_obtenida']}")

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(filas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetalle completo guardado en {SALIDA.relative_to(BASE)}")


if __name__ == "__main__":
    main()
