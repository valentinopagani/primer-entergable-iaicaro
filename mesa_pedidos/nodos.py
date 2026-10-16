import json
import random
import re
from datetime import datetime
from pathlib import Path

from langgraph.types import interrupt

from mesa_pedidos.estado import EstadoMesa, ClasificacionMensaje, PedidoExtraido, DecisionHumana
from mesa_pedidos.llm import crear_llm
from mesa_pedidos.retriever import responder as responder_rag
from mesa_pedidos import reglas
from mesa_pedidos.tools import buscar_cliente, consultar_producto, verificar_credito, cotizar_flete

llm = crear_llm()

clasificador = llm.with_structured_output(ClasificacionMensaje)
extractor = llm.with_structured_output(PedidoExtraido)

PATRON_CUIT = re.compile(r"\b\d{2}-?\d{8}-?\d\b")
BASE = Path(__file__).resolve().parent.parent
SALIDAS = BASE / "artefactos" / "salidas"


def _guardar(nombre: str, data: dict) -> str:
    SALIDAS.mkdir(parents=True, exist_ok=True)
    p = SALIDAS / nombre
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# 1) cargar_cliente
# ---------------------------------------------------------------------------
def cargar_cliente(state: EstadoMesa) -> dict:
    cuit = state.get("cuit")
    if not cuit:
        match = PATRON_CUIT.search(state.get("mensaje", ""))
        cuit = match.group() if match else None
    cliente = buscar_cliente(cuit) if cuit else None
    if cliente is None:
        return {"cuit": cuit, "cliente": None,
                "errores": [f"CUIT {cuit} no esta en clientes.csv"],
                "log": [f"cliente no encontrado: {cuit}"]}
    return {"cuit": cuit, "cliente": cliente,
            "log": [f"cliente resuelto: {cuit}"]}


# ---------------------------------------------------------------------------
# 2) clasificar — LLM + regla intencion_efectiva
# ---------------------------------------------------------------------------
def clasificar(state: EstadoMesa) -> dict:
    prompt = (
        "Sos el clasificador de la Mesa de Pedidos de Ferreteria Industrial Andina. "
        "Clasifica en 'pedido' | 'consulta' | 'derivar' y da confianza 0-1.\n"
        "- 'pedido': pide mercaderia concreta con productos y cantidades "
        "(\"mandame...\", \"necesito N unidades\", \"quiero 20 de...\").\n"
        "- 'consulta': pregunta de politica/operativa (flete, devoluciones, garantia, "
        "precios, presupuestos) o especificaciones de producto. Preguntar por limites, "
        "precios o disponibilidad (\"cuantos me pueden mandar\", \"cuanto sale\") es "
        "'consulta' aunque mencione cantidades: pedido requiere intencion de compra.\n"
        "- 'derivar': asesoramiento sin producto definido, que lo llamen, reclamos de "
        "envios ya hechos, proveedores.\n"
        "Ejemplos: \"viene con bateria?\" -> consulta; "
        "\"me equivoque de talle, lo puedo devolver?\" -> consulta; "
        "\"mandame 20 unidades\" -> pedido; \"llego con faltantes\" -> derivar.\n\n"
        f"Mensaje: {state['mensaje']}"
    )
    r = clasificador.invoke(prompt)
    inten = r.intencion or r.tipo
    clasif = {"intencion": inten, "tipo": r.tipo, "confianza": float(r.confianza or 1.0),
              "motivo": r.motivo, "area_sugerida": r.area_sugerida}
    intencion, nota = reglas.intencion_efectiva(clasif, state.get("cliente"))
    detalle = f"clasificacion: {inten} ({clasif['confianza']:.2f}) - {r.motivo}"
    if nota:
        detalle += f" -> {nota}"
    return {"clasificacion": clasif, "intencion": intencion,
            "motivo_clasificacion": r.motivo, "log": [detalle]}


# ---------------------------------------------------------------------------
# 3) responder_consulta
# ---------------------------------------------------------------------------
def responder_consulta(state: EstadoMesa) -> dict:
    r = responder_rag(state["mensaje"], state.get("cliente"))
    return {"respuesta": r["respuesta"], "fuentes": r["fuentes"],
            "citas": r["fuentes"], "ruta": "responder_consulta",
            "log": ["consulta respondida con RAG (densa + cita)"]}


# ---------------------------------------------------------------------------
# 4) extraer_pedido — sin catalogo en prompt
# ---------------------------------------------------------------------------
def extraer_pedido(state: EstadoMesa) -> dict:
    prompt = (
        "Extrae el pedido mayorista del mensaje. Para cada item: sku (codigo XX-0000 "
        "SOLO si el cliente lo escribio literal, sino null), descripcion textual, cantidad entera. "
        "Ademas: descuento_solicitado (% o null), pago_anticipado (true solo si dice que paga "
        "antes del envio), destino (localidad si la menciona), observaciones.\n\n"
        f"Mensaje: {state['mensaje']}"
    )
    r = extractor.invoke(prompt)
    pedido = r.model_dump()
    items = [i if isinstance(i, dict) else i for i in pedido.get("items", [])]
    return {"pedido": pedido, "items": items,
            "descuento_solicitado": pedido.get("descuento_solicitado"),
            "destino": pedido.get("destino"),
            "log": [f"pedido extraido: {len(items)} items"
                     + (" | pago anticipado" if pedido.get("pago_anticipado") else "")]}


# ---------------------------------------------------------------------------
# 5) validar_pedido — tools + reglas, SIN LLM
# ---------------------------------------------------------------------------
def _resolver_items(pedido: dict):
    resueltos, no_resueltos, inferidos = [], [], []
    for item in pedido.get("items", []):
        info = consultar_producto.invoke({"sku": item.get("sku"),
                                          "descripcion": item.get("descripcion") or "",
                                          "cantidad": int(item.get("cantidad", 1))})
        if info.get("encontrado"):
            resueltos.append({**info, "cantidad": int(item.get("cantidad", 1))})
        elif len(info.get("sugerencias", [])) == 1:
            sku = info["sugerencias"][0]["sku"]
            r2 = consultar_producto.invoke({"sku": sku, "cantidad": int(item.get("cantidad", 1))})
            resueltos.append({**r2, "cantidad": int(item.get("cantidad", 1))})
            inferidos.append(sku)
        else:
            no_resueltos.append({"descripcion": item.get("descripcion"), "sku": item.get("sku"),
                                 "sugerencias": info.get("sugerencias", [])})
    return resueltos, no_resueltos, inferidos


def _calcular(cliente: dict, items: list[dict], pago_anticipado: bool,
              descuento_solicitado=None, descuento_autorizado=None) -> dict:
    valor = reglas.valorizar([{"sku": i["sku"], "nombre": i["nombre"], "marca": i.get("marca", ""),
                               "rubro": i.get("rubro", ""), "cantidad": i["cantidad"],
                               "precio_lista": i["precio_lista"],
                               "alicuota_iva": i.get("alicuota_iva", 21.0),
                               "stock": i.get("stock", 0), "deposito": i.get("deposito", "")}
                              for i in items],
                             cliente, descuento_solicitado, pago_anticipado, descuento_autorizado)
    credito = verificar_credito.invoke({"cuit": cliente["cuit"], "monto_neto": valor["neto"],
                                        "pago_anticipado": pago_anticipado})
    aerosoles = sum(i["cantidad"] for i in items if "aerosol" in reglas.normalizar_texto(i.get("nombre", "")))
    flete = cotizar_flete.invoke({"provincia": cliente.get("provincia", ""),
                                  "localidad": cliente.get("localidad", ""),
                                  "monto_neto": valor["neto"],
                                  "pago_anticipado": pago_anticipado,
                                  "unidades_aerosol": aerosoles})
    observaciones = list(valor.get("notas", [])) + [flete.get("detalle", "")]
    total = valor["neto"] + valor["iva"] + flete.get("costo", 0) * 1.21
    return {"valor": valor, "credito": credito, "flete": flete,
            "observaciones": observaciones, "total": round(total, 2)}


def validar_pedido(state: EstadoMesa) -> dict:
    cliente = state.get("cliente")
    pedido = state.get("pedido") or {}
    if cliente is None:
        return {"decision": "derivar",
                "motivo_derivacion": "No se pudo identificar al cliente (CUIT ausente o inexistente)",
                "log": ["validacion: sin cliente -> derivar"]}
    pago_anticipado = bool(pedido.get("pago_anticipado"))
    items, no_resueltos, inferidos = _resolver_items(pedido)
    calc = _calcular(cliente, items, pago_anticipado, pedido.get("descuento_solicitado"))
    valor = calc["valor"]
    if pedido.get("destino") and reglas.normalizar_texto(cliente.get("localidad", "")) not in reglas.normalizar_texto(pedido["destino"]):
        calc["observaciones"].append(f"Destino indicado '{pedido['destino']}'; flete cotizado a ficha ({cliente.get('localidad')})")
    ruta, motivo, alertas = reglas.decidir_ruta(
        cliente, valor, calc["credito"], calc["flete"], no_resueltos, inferidos,
        pedido.get("descuento_solicitado"))
    validacion = {"ruta": ruta, "motivo": motivo, "alertas": alertas,
                  "observaciones": calc["observaciones"],
                  "renglones": valor["renglones"], "no_resueltos": no_resueltos,
                  "bruto": valor["bruto"], "descuento_volumen_pct": valor["descuento_volumen_pct"],
                  "descuento_aplicado_pct": valor["descuento_aplicado_pct"],
                  "descuento_financiero": valor["descuento_financiero"],
                  "neto": valor["neto"], "iva": valor["iva"], "flete": calc["flete"]["costo"],
                  "total_con_iva": calc["total"], "credito": calc["credito"],
                  "cotizacion_flete": calc["flete"], "pago_anticipado": pago_anticipado}
    mapa = {"confirmar_pedido": "confirmar", "aprobacion_humana": "humano",
            "rechazar_pedido": "rechazar", "derivar_humano": "derivar"}
    decision = mapa[ruta]
    # pedido legacy enriquecido para payload/correr compat
    pedido_leg = {"renglones": valor["renglones"], "neto": valor["neto"],
                  "descuento_total": valor["descuento_aplicado_pct"],
                  "descuento_aplicado_pct": valor["descuento_aplicado_pct"],
                  "tope_descuento": 20.0, "iva": valor["iva"], "total": calc["total"],
                  "total_con_iva": calc["total"], "credito": calc["credito"],
                  "flete": calc["flete"]["costo"]}
    out = {"validacion": validacion, "pedido": pedido_leg, "alertas": alertas,
           "decision": decision,
           "log": [f"validacion: neto ${valor['neto']:,.0f}, descuento "
                   f"{valor['descuento_aplicado_pct']:.1f}%, decision={decision} ({len(alertas)} alertas)"]}
    if decision == "derivar":
        out["motivo_derivacion"] = motivo
    return out


# ---------------------------------------------------------------------------
# 6) aprobacion_humana — aprobar / editar / rechazar
# ---------------------------------------------------------------------------
def _payload(state: EstadoMesa) -> dict:
    v, c = state["validacion"], state["cliente"]
    cred = v.get("credito", {})
    return {"mensaje_id": state.get("mensaje_id", ""), "accion": "aprobar_pedido",
            "razon_social": c["razon_social"], "cuit": c["cuit"], "categoria": c.get("categoria"),
            "mora_dias": c.get("mora_dias"),
            "limite_credito": cred.get("limite_credito"), "saldo_actual": cred.get("saldo_actual"),
            "credito_disponible": cred.get("credito_disponible"),
            "neto": v["neto"], "descuento_total": v.get("descuento_aplicado_pct", 0),
            "descuento_aplicado_pct": v.get("descuento_aplicado_pct", 0),
            "iva": v["iva"], "total": v.get("total_con_iva", 0),
            "total_con_iva": v.get("total_con_iva", 0),
            "motivo": v.get("motivo", ""), "alertas": v.get("alertas", []),
            "renglones": v.get("renglones", []),
            "texto_original": state.get("mensaje", ""),
            "options": ["aprobar", "editar", "rechazar"],
            "opciones": ["aprobar", "editar", "rechazar"]}


def _aplicar_edicion(state: EstadoMesa, decision: DecisionHumana) -> dict:
    v, cliente = state["validacion"], state["cliente"]
    items = []
    for r in v["renglones"]:
        cant = decision.items_editados.get(r["sku"], r["cantidad"])
        if cant > 0:
            items.append({"sku": r["sku"], "nombre": r["nombre"], "marca": r.get("marca", ""),
                          "rubro": r.get("rubro", ""), "cantidad": cant,
                          "precio_lista": r.get("precio_unitario", 0),
                          "alicuota_iva": r.get("alicuota_iva", 21.0),
                          "stock": r.get("stock_disponible", r.get("stock", 0)),
                          "deposito": r.get("deposito", "")})
    calc = _calcular(cliente, items, v.get("pago_anticipado", False),
                     descuento_autorizado=decision.descuento_autorizado)
    valor = calc["valor"]
    return {"ruta": "confirmar_pedido",
            "motivo": f"Editado y autorizado por {decision.autorizado_por}: {decision.razon}".strip(": "),
            "alertas": v.get("alertas", []),
            "observaciones": calc["observaciones"] + ["Pedido editado por Gerencia antes de confirmar"],
            "renglones": valor["renglones"], "bruto": valor["bruto"],
            "descuento_volumen_pct": valor["descuento_volumen_pct"],
            "descuento_aplicado_pct": valor["descuento_aplicado_pct"],
            "descuento_financiero": valor["descuento_financiero"],
            "neto": valor["neto"], "iva": valor["iva"], "flete": calc["flete"]["costo"],
            "total_con_iva": calc["total"], "credito": calc["credito"],
            "cotizacion_flete": calc["flete"], "pago_anticipado": v.get("pago_anticipado", False)}


def aprobacion_humana(state: EstadoMesa) -> dict:
    crudo = interrupt(_payload(state))
    if isinstance(crudo, str):
        crudo = {"tipo": crudo}
    elif isinstance(crudo, dict) and "aprobado" in crudo and "tipo" not in crudo:
        crudo = {"tipo": "aprobar" if crudo.get("aprobado") else "rechazar",
                 "razon": crudo.get("motivo", "")}
    try:
        decision = DecisionHumana(**(crudo or {}))
    except Exception:
        decision = DecisionHumana(tipo="rechazar", razon=f"Decision ilegible: {crudo!r}")
    salida: dict = {"decision_humana": decision.model_dump(),
                    "aprobacion": {"aprobado": decision.tipo in ("aprobar", "editar"),
                                   "motivo": decision.razon, "tipo": decision.tipo}}
    detalle = f"{decision.tipo.upper()} por {decision.autorizado_por}"
    if decision.razon:
        detalle += f" - {decision.razon}"
    if decision.tipo == "editar":
        salida["validacion"] = _aplicar_edicion(state, decision)
        # sincronizar pedido legacy con la edicion
        vv = salida["validacion"]
        salida["pedido"] = {"renglones": vv["renglones"], "neto": vv["neto"],
                             "descuento_total": vv["descuento_aplicado_pct"],
                             "descuento_aplicado_pct": vv["descuento_aplicado_pct"],
                             "iva": vv["iva"], "total": vv["total_con_iva"],
                             "total_con_iva": vv["total_con_iva"],
                             "credito": vv["credito"], "flete": vv["flete"]}
        salida["decision"] = "confirmar"
        detalle += f" | nuevo neto ${vv['neto']:,.0f}"
    elif decision.tipo == "aprobar":
        salida["decision"] = "confirmar"
    else:
        salida["decision"] = "rechazar"
        salida["motivo_rechazo"] = decision.razon or "rechazado por Gerencia Comercial"
    salida["log"] = [f"aprobacion humana: {detalle}"]
    return salida


# ---------------------------------------------------------------------------
# 7-9) hojas
# ---------------------------------------------------------------------------
def _redactar(prompt: str, fallback: str) -> str:
    try:
        return llm.invoke(prompt).content.strip() or fallback
    except Exception:
        return fallback


def confirmar_pedido(state: EstadoMesa) -> dict:
    v, c = state["validacion"], state["cliente"]
    dec = state.get("decision_humana") or {}
    numero = f"PED-{datetime.now().year}-{random.randint(1000, 9999)}"
    resumen = {"numero": numero, "mensaje_id": state.get("mensaje_id", ""),
               "cuit": c["cuit"], "razon_social": c["razon_social"], "estado": "confirmado",
               "renglones": v["renglones"], "neto": v["neto"],
               "descuento_aplicado_pct": v.get("descuento_aplicado_pct", 0),
               "flete": v.get("flete", 0), "total_con_iva": v.get("total_con_iva", 0),
               "autorizado_por": dec.get("autorizado_por"), "motivo": v.get("motivo", "")}
    mid = state.get("mensaje_id") or "sin-id"
    _guardar(f"pedidos/{mid}.json", resumen)
    fb = (f"Pedido {numero} confirmado para {c['razon_social']}. "
          f"Neto ${v['neto']:,.2f} + IVA = total ${v.get('total_con_iva', 0):,.2f}.")
    resp = _redactar(f"Redacta confirmacion breve rioplatense con estos datos:\n{fb}\nPedido original: {state.get('mensaje','')}", fb)
    return {"resumen": resumen, "respuesta": resp, "ruta": "confirmar_pedido",
            "decision": "confirmar", "log": [f"pedido confirmado {numero}"]}


def rechazar_pedido(state: EstadoMesa) -> dict:
    v, c = state.get("validacion") or {}, state.get("cliente") or {}
    dec = state.get("decision_humana") or {}
    motivo = state.get("motivo_rechazo")
    if not motivo:
        if dec.get("tipo") == "rechazar":
            motivo = f"Gerencia Comercial no autorizo el pedido. {dec.get('razon', '')}".strip()
        else:
            motivo = v.get("motivo") or "; ".join((state.get("alertas") or [])) or "rechazado"
    numero = f"PED-{datetime.now().year}-{random.randint(1000, 9999)}"
    resumen = {"numero": numero, "mensaje_id": state.get("mensaje_id", ""),
               "cuit": c.get("cuit", ""), "razon_social": c.get("razon_social", ""),
               "estado": "rechazado", "renglones": v.get("renglones", []),
               "neto": v.get("neto", 0), "motivo": motivo}
    mid = state.get("mensaje_id") or "sin-id"
    _guardar(f"pedidos/{mid}.json", resumen)
    fb = f"Pedido rechazado: {motivo}."
    resp = _redactar(f"Redacta rechazo breve con alternativa (pago anticipado/regularizar saldo):\n{fb}", fb)
    return {"resumen": resumen, "respuesta": fb if resp == fb else resp,
            "ruta": "rechazar_pedido", "decision": "rechazar",
            "log": [f"pedido rechazado — {motivo[:110]}"]}


_AREAS = {"mesa_de_pedidos": "Mesa de Pedidos", "reclamos_y_logistica": "Reclamos y Logistica",
          "compras_proveedores": "Compras", "asesoramiento_tecnico": "Asesoramiento Tecnico",
          "administracion": "Administracion"}


def derivar_humano(state: EstadoMesa) -> dict:
    clasif = state.get("clasificacion") or {}
    validacion = state.get("validacion") or {}
    if validacion.get("ruta") == "derivar_humano":
        origen, area, motivo = "validacion", "mesa_de_pedidos", validacion.get("motivo", "")
    else:
        origen = "clasificacion"
        area = (clasif.get("area_sugerida") if isinstance(clasif, dict) else None) or "mesa_de_pedidos"
        if state.get("motivo_derivacion"):
            motivo = state["motivo_derivacion"]
        elif isinstance(clasif, dict):
            motivo = clasif.get("motivo", "Mensaje fuera del alcance")
            if state.get("intencion") == "derivar" and clasif.get("intencion") != "derivar":
                motivo += " (derivado por regla: confianza baja o cliente desconocido)"
        elif clasif == "derivar":
            motivo = state.get("motivo_clasificacion") or "reclamo o asesoramiento"
        else:
            motivo = "requiere intervencion humana"
    numero = f"TCK-{datetime.now().year}-{random.randint(1000, 9999)}"
    ticket = {"numero": numero, "mensaje_id": state.get("mensaje_id", ""),
              "area": _AREAS.get(area, area), "motivo": motivo, "origen": origen,
              "resumen": (state.get("mensaje") or "")[:160],
              "cliente": (state.get("cliente") or {}).get("razon_social")}
    mid = state.get("mensaje_id") or "sin-id"
    _guardar(f"tickets/{mid}.json", ticket)
    return {"ticket": ticket, "respuesta": f"Derivado a una persona ({ticket['area']}): {motivo}. Caso {numero}.",
            "ruta": "derivar_humano", "log": [f"derivado a humano {numero} -> {ticket['area']} (origen: {origen})"]}
