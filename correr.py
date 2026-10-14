"""correr.py — procesa UN mensaje de la bandeja y maneja la aprobación humana (4.5).

El grafo se compila con un SqliteSaver sobre archivo, con un thread_id por
mensaje. Así el proceso puede frenar en la aprobación de Gerencia, cerrarse, y
retomarse después (en otra terminal, otro día) con Command(resume=...).

Uso:
    # 1) Arranca el mensaje. Si cae en humano, frena, muestra el payload y sale.
    python correr.py MSG-016

    # 2) Retoma esa aprobación pendiente (mismo thread_id, otra corrida):
    python correr.py MSG-016 --resume --decision aprobar
    python correr.py MSG-016 --resume --decision rechazar --motivo "excede la política"
"""

import argparse
import json
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from mesa_pedidos.grafo import construir_grafo

BASE = Path(__file__).resolve().parent
BANDEJA = BASE / "datos" / "bandeja.jsonl"
DB_CHECKPOINTS = BASE / "artefactos" / "checkpoints.sqlite"


def leer_mensaje(mensaje_id: str) -> dict:
    """Busca el mensaje por ID en bandeja.jsonl."""
    with open(BANDEJA, encoding="utf-8") as f:
        for linea in f:
            if not linea.strip():
                continue
            msg = json.loads(linea)
            if msg["id"] == mensaje_id:
                return msg
    raise SystemExit(f"No encontré el mensaje '{mensaje_id}' en {BANDEJA.name}")


def construir_grafo_con_memoria():
    """Compila el grafo con el SqliteSaver sobre archivo.

    La conexión tiene que quedar viva mientras el grafo corre, por eso se
    devuelve junto con el grafo y se cierra al final de main().
    """
    DB_CHECKPOINTS.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_CHECKPOINTS, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return construir_grafo(checkpointer=checkpointer), conn


def mostrar_payload(payload: dict) -> None:
    """Muestra a la persona todo lo que necesita para decidir."""

    def plata(valor) -> str:
        return f"${valor:,.0f}" if isinstance(valor, (int, float)) else "N/D"

    print("\n" + "=" * 72)
    print("APROBACIÓN HUMANA REQUERIDA — Gerencia Comercial")
    print("=" * 72)
    print(f"Cliente:  {payload['razon_social']} (CUIT {payload['cuit']}, cat. {payload['categoria']})")
    print(
        f"Crédito:  disponible {plata(payload['credito_disponible'])} | "
        f"límite {plata(payload['limite_credito'])} | saldo {plata(payload['saldo_actual'])}"
    )
    print(
        f"Pedido:   neto ${payload['neto']:,.2f} + IVA ${payload['iva']:,.2f} "
        f"= total ${payload['total']:,.2f} (descuento {payload['descuento_total']:.1f}%)"
    )

    print("\nAlertas:")
    for alerta in payload["alertas"] or ["(ninguna)"]:
        print(f"  - {alerta}")

    print("\nRenglones:")
    for r in payload["renglones"]:
        print(
            f"  {r['sku']:>8}  {r['nombre'][:36]:<36} "
            f"{r['cantidad']:>8g} x ${r['precio_unitario']:>12,.2f} = ${r['subtotal']:>14,.2f}"
        )
    print("=" * 72)


def main() -> None:
    import sys
    # Las respuestas del LLM traen espacios finos y guiones unicode que la
    # consola Windows (cp1252) no puede imprimir: forzar UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Procesa un mensaje de la bandeja.")
    parser.add_argument("mensaje_id", help="ID del mensaje, ej: MSG-016")
    parser.add_argument("--resume", action="store_true", help="Retomar una aprobación pendiente")
    parser.add_argument(
        "--decision", choices=["aprobar", "rechazar"], default="aprobar",
        help="Decisión de Gerencia al retomar (por defecto aprobar)",
    )
    parser.add_argument("--motivo", default="", help="Motivo de la decisión (recomendado al rechazar)")
    args = parser.parse_args()

    grafo, conn = construir_grafo_con_memoria()
    config = {"configurable": {"thread_id": f"msg-{args.mensaje_id}"}}

    try:
        if args.resume:
            decision = {
                "aprobado": args.decision == "aprobar",
                "motivo": args.motivo,
            }
            out = grafo.invoke(Command(resume=decision), config=config)
            print(f"\nRespuesta: {out.get('respuesta')}")
            print(f"Decisión final: {out.get('decision')} | aprobación={out.get('aprobacion')}")
        else:
            msg = leer_mensaje(args.mensaje_id)
            estado_inicial = {"mensaje": msg["texto"], "cuit": msg["cuit"]}
            out = grafo.invoke(estado_inicial, config=config)

            if "__interrupt__" in out:
                mostrar_payload(out["__interrupt__"][0].value)
                print("\nEl proceso frenó y el estado quedó guardado en el checkpointer.")
                print("Para retomar (puede ser otra terminal):")
                print(f"  python correr.py {args.mensaje_id} --resume --decision aprobar")
                print(f"  python correr.py {args.mensaje_id} --resume --decision rechazar --motivo \"...\"")
            else:
                print(f"\nResuelto sin aprobación. Clasificación: {out.get('clasificacion')}")
                print(f"Respuesta: {out.get('respuesta')}")

        print("\nLog:")
        for linea in out.get("log", []):
            print(f"  · {linea}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
