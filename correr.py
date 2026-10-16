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
        f"= total ${payload.get('total_con_iva', payload.get('total', 0)):,.2f} "
        f"(descuento {payload.get('descuento_aplicado_pct', payload.get('descuento_total', 0)):.1f}%)"
    )
    if payload.get("motivo"):
        print(f"Motivo:   {payload['motivo']}")
    if payload.get("texto_original"):
        print(f"Original: {payload['texto_original'][:160]}")

    print("\nAlertas:")
    for alerta in payload["alertas"] or ["(ninguna)"]:
        print(f"  - {alerta}")

    print("\nRenglones:")
    for r in payload["renglones"]:
        print(
            f"  {r.get('sku', '?'):>8}  {r.get('nombre', '')[:36]:<36} "
            f"{r.get('cantidad', 0):>8g} x ${r.get('precio_unitario', 0):>12,.2f} = "
            f"${r.get('subtotal_neto', r.get('subtotal', 0)):>14,.2f}"
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
    parser.add_argument("mensaje_id", nargs="?", help="ID del mensaje, ej: MSG-016")
    parser.add_argument("--resume", action="store_true", help="Retomar una aprobación pendiente")
    parser.add_argument(
        "--decision", choices=["aprobar", "editar", "rechazar"], default="aprobar",
        help="Decisión de Gerencia al retomar (por defecto aprobar)",
    )
    parser.add_argument("--motivo", default="", help="Motivo de la decisión (recomendado al rechazar)")
    parser.add_argument("--items", default="", help='Editar: JSON {"SKU": nueva_cantidad}')
    parser.add_argument("--descuento", type=float, default=None, help="Editar: descuento autorizado %")
    parser.add_argument("--pendientes", action="store_true", help="Lista threads interrumpidos")
    parser.add_argument("--todos", action="store_true", help="Corre los 20 mensajes")
    parser.add_argument("--texto", default="", help="Mensaje libre (requiere --cuit)")
    parser.add_argument("--cuit", default="", help="CUIT para --texto")
    args = parser.parse_args()

    grafo, conn = construir_grafo_con_memoria()
    config = {"configurable": {"thread_id": f"msg-{args.mensaje_id}"}}

    try:
        if args.pendientes:
            from langgraph.checkpoint.sqlite import SqliteSaver as _S
            print("Threads con estado guardado:")
            for t in conn.execute(
                    "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id").fetchall():
                print(f"  - {t[0]}")
            return
        if args.todos:
            for m in [leer_mensaje(f"MSG-{i:03d}") for i in range(1, 21)]:
                try:
                    out = grafo.invoke({"mensaje": m["texto"], "mensaje_id": m["id"], "cuit": m["cuit"]},
                                       config={"configurable": {"thread_id": f"msg-{m['id']}"}})
                    estado = "INTERRUMPIDO" if "__interrupt__" in out else out.get("ruta", out.get("decision"))
                    print(f"{m['id']}: {estado}")
                except Exception as e:
                    print(f"{m['id']}: ERROR {e}")
            return
        if args.texto:
            if not args.cuit:
                raise SystemExit("--texto requiere --cuit")
            mensaje_id = args.mensaje_id or "manual"
            estado_inicial = {"mensaje": args.texto, "mensaje_id": mensaje_id, "cuit": args.cuit}
            thread = f"msg-{mensaje_id}"
        elif not args.mensaje_id:
            raise SystemExit("Indica mensaje_id, --todos, --pendientes o --texto")
        else:
            msg = leer_mensaje(args.mensaje_id)
            estado_inicial = {"mensaje": msg["texto"], "mensaje_id": msg["id"], "cuit": msg["cuit"]}
            thread = f"msg-{args.mensaje_id}"
        config = {"configurable": {"thread_id": thread}}

        if args.resume:
            if args.decision == "editar":
                items_edit = json.loads(args.items or "{}")
                decision = {"tipo": "editar", "razon": args.motivo,
                            "items_editados": items_edit,
                            "descuento_autorizado": args.descuento}
            else:
                decision = {"tipo": "aprobar" if args.decision == "aprobar" else "rechazar",
                            "razon": args.motivo}
            out = grafo.invoke(Command(resume=decision), config=config)
            print(f"\nRespuesta: {out.get('respuesta')}")
            print(f"Decisión final: {out.get('ruta', out.get('decision'))} | humana={out.get('decision_humana', out.get('aprobacion'))}")
        else:
            out = grafo.invoke(estado_inicial, config=config)

            if "__interrupt__" in out:
                mostrar_payload(out["__interrupt__"][0].value)
                print("\nEl proceso frenó y el estado quedó guardado en el checkpointer.")
                print("Para retomar (puede ser otra terminal):")
                print(f"  python correr.py {args.mensaje_id or 'manual'} --resume --decision aprobar")
                print(f"  python correr.py {args.mensaje_id or 'manual'} --resume --decision rechazar --motivo \"...\"")
                print(f"  python correr.py {args.mensaje_id or 'manual'} --resume --decision editar --items '{{\"SKU\": 5}}'")
            else:
                print(f"\nResuelto sin aprobación. Ruta: {out.get('ruta', out.get('decision'))}")
                print(f"Respuesta: {out.get('respuesta')}")

        print("\nLog:")
        for linea in out.get("log", []):
            print(f"  · {linea}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
