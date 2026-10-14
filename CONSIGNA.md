# Desafío 1 — Mesa de Pedidos

**Diplomatura Universitaria de Formación Continua en IA Engineering**
ICARO · Universidad Nacional de Córdoba · FCEFyN ·

---

## 1. La situación

Ferretería Industrial Andina S.R.L. es una distribuidora mayorista de Córdoba capital. Vende insumos
industriales —abrasivos, herramienta eléctrica, elementos de seguridad, fijaciones— a ferreterías,
talleres metalúrgicos y constructoras de medio país. Dos depósitos, doscientos clientes con cuenta
corriente, y ningún portal: los pedidos entran por WhatsApp y por mail.

Llega un mensaje que dice *"che, mandame 200 discos de corte y 50 de desbaste, a Villa María como
siempre"* y alguien lo tiene que convertir en un pedido. Ese alguien es Vanesa, y lo primero que hace
es decidir qué es ese mensaje: ¿pide mercadería, pregunta algo, o es un reclamo que va para otro lado?

Si es una pregunta —*"cuánto sale el flete a Neuquén"*, *"puedo devolver unos zapatos que compré
mal"*— la respuesta está en alguno de los cuatro PDF de la empresa, y nunca se acuerda en cuál. Si es
un pedido, busca cada producto, chequea stock, arma el total, aplica el descuento del cliente y —esto
es lo importante— **mira si el cliente tiene crédito disponible**. Si el pedido se pasa del límite, o
el cliente está en mora, o pide un descuento por encima de lo que ella puede dar, **no confirma
nada**: lo manda a la Gerencia Comercial y espera.

Hace esto cuarenta veces por día. El 70% son casos rutinarios; el otro 30% son los que importan, y
son justo los que atiende peor, porque llega al final del día cansada de copiar SKUs.

**Lo que vas a construir es el sistema que hace el trabajo rutinario y sabe cuándo tiene que llamar a
una persona.** No es un chatbot: clasifica, busca en los documentos de la empresa, arma el pedido
contra los datos reales, aplica las reglas comerciales, y **frena y pregunta** cuando la decisión no
le corresponde. Un sistema que puede confirmar un pedido de doce millones de pesos sin que nadie lo
mire es un sistema que no se pone en producción.

---

## 2. Qué hay que construir

Un grafo de LangGraph que procese un mensaje de punta a punta: nueve nodos, tres puntos de ruteo,
cinco hojas. Arrancás con esta carpeta y nada más, así que el armado del proyecto es parte del
trabajo. Esta es la forma a la que tenés que llegar:

```
                        START
                          │
                   [cargar_cliente]          ← resuelve el CUIT contra los datos
                          │
                     [clasificar]            ← LLM + structured output
                          │
              ┌───────────┼───────────┐
        (consulta)    (pedido)    (derivar)
              │           │           │
   [responder_consulta]   │   [derivar_humano]
         ↑ RAG            │           │
              │     [extraer_pedido]  │      ← LLM + structured output
             END          │          END
                   [validar_pedido]          ← tools + reglas, SIN LLM
                          │
        ┌────────┬────────┴────────┬──────────┐
  (confirmar) (humano)      (rechazar)   (derivar)
        │        │                │           │
        │ [aprobacion_humana]     │           │
        │    │        │           │           │
        │ aprueba   rechaza       │           │
        │    │        │           │           │
   [confirmar_pedido] │           │           │
        │             └──→ [rechazar_pedido]  │
       END                        │    [derivar_humano]
                                 END          │
                                             END
```

### El vocabulario

| Término | Qué es, literalmente |
|---|---|
| **Nodo** | Una función que recibe el estado y devuelve un `dict` con los campos que cambió. No devuelve el estado entero y no llama al nodo siguiente: hace su parte y termina. |
| **El estado** (*state*) | Un diccionario tipado (`TypedDict`) que viaja de nodo en nodo y es el único canal por el que los nodos se pasan información. Lo que no quedó escrito ahí, después no se puede auditar: un `print` no es estado. |
| **Reducer** | La función que decide qué pasa cuando un nodo escribe un campo que ya tenía algo. Sin reducer el valor nuevo pisa al viejo; con `Annotated[list[str], add]` se concatenan, y por eso el `log` termina con una línea por nodo. |
| **Arista condicional** | Una flecha cuyo destino no se conoce hasta que se ejecuta. Son dos cosas juntas: una función *router* que lee el estado y devuelve un string, y un mapa que traduce cada string a un nodo. El router no decide nada de negocio: traduce una decisión que ya tomó un nodo. |
| **Checkpointer** y **`thread_id`** | El checkpointer guarda una copia del estado después de cada nodo, indexada por `thread_id`; `SqliteSaver` la escribe en un archivo y sobrevive al reinicio. El `thread_id` es el string con el que indexa: mismo `thread_id`, misma historia. Acá va uno por mensaje. |
| **Herramienta** (*tool*) | Una función normal con tres cosas encima: el decorador `@tool`, un `args_schema` de Pydantic que declara qué recibe, y un docstring que declara qué hace. Devuelve siempre algo serializable, nunca una excepción sin atrapar. |
| **Human-in-the-loop** | Un punto del grafo donde el programa se frena, guarda en el checkpoint lo que una persona necesita para decidir, y termina. La diferencia con un `input()` es esa: el proceso **termina**, y se retoma después —en otra terminal, otro día— con un `Command(resume=...)`. Se escribe con `interrupt()`. |
| **Structured output** | Pedirle al LLM que conteste con una forma declarada de antemano en un modelo de Pydantic. La librería valida la respuesta y te devuelve un objeto de Python, o falla ruidosamente. Es lo contrario de parsear JSON con `json.loads` adentro de un `try`. |
| **Golden set** | Un archivo con casos de prueba y la respuesta correcta de cada uno, escrita a mano antes de medir nada. Sin él, "funciona" es una opinión. |
| **Búsqueda densa / BM25** | **Densa**: la consulta se convierte en vector y se traen los fragmentos con el vector más parecido; encuentra textos que *significan* algo parecido aunque no compartan una palabra. **BM25**: cuenta palabras exactas ponderadas por cuán raras son; encuentra el `HE-4330` literal. |

---

## 3. Los datos que te damos

Están en esta carpeta. Son cuatro fuentes y **no hay que inventar ni salir a buscar nada más**.

1. **Las políticas de la empresa** — `politicas/`: 4 PDF, 10 páginas, con las condiciones comerciales,
   la política de devoluciones, logística y fletes, y las fichas técnicas. Es el corpus del RAG: acá
   están las reglas que el sistema tiene que leer en vez de tenerlas hardcodeadas.
2. **Los datos de la empresa** — `catalogo.csv` (40 productos con precio, IVA, stock y depósito) y
   `clientes.csv` (12 clientes con CUIT, límite de crédito, saldo, mora y descuento habitual). Son
   los datos duros que no se pueden inventar: de acá sale el precio y el crédito disponible.
3. **La bandeja de entrada** — `bandeja.jsonl`: los 20 mensajes de clientes, 8 consultas, 9 pedidos y
   3 para derivar. Es la entrada del sistema, lo que le llega a Vanesa.
4. **Los casos de prueba** — `golden_set.json`: los mismos 20 mensajes con la respuesta correcta
   escrita a mano (la rama, los items y el dato clave). Es la referencia contra la que vas a medir.
   **No lo mires hasta tener el grafo andando**, o vas a ajustar el sistema a la respuesta.

Los CSV abren con Excel: miralos antes de empezar. Es ficticio pero consistente —los CUIT tienen
dígito verificador válido, las provincias coinciden con las zonas de flete, y los saldos están
puestos para que cada caso del golden set caiga en la rama que dice que cae—.

---

## 4. Las seis cosas que tenés que hacer

Son las seis piezas que el sistema necesita para funcionar. Si falta una, no hay sistema: hay un pedazo.

### 4.1 Indexar los PDF y responder citando la fuente *(Clase 4)*

Indexá los 4 PDF con embeddings y **persistí el índice**: reindexar en cada corrida es tirar plata. La
respuesta tiene que citar **documento y página**, para que alguien la pueda verificar. Si el contexto
no alcanza, el sistema lo dice y no completa. **La búsqueda densa alcanza.**

### 4.2 Dos tools que consulten los datos de verdad *(Clase 6)*

`consultar_producto` y `verificar_credito`, con `args_schema` de Pydantic y docstring legible. "De
verdad" quiere decir que leen los CSV y devuelven datos duros: una tool que devuelve un string fijo no
cuenta. Acá las tools no las llama un agente decidiendo solo —las llaman nodos, en el orden que
definiste vos—, y esa es la diferencia entre la Clase 6 y la Clase 7.

### 4.3 Un grafo con estado *(Clase 7)*

`StateGraph` con un `TypedDict` bien pensado, y el campo de log **con su reducer**
(`Annotated[list[str], add]`). Si tu log termina con una sola línea, te faltó eso. La clasificación y
la extracción del pedido salen del LLM con **structured output**: definí un modelo de Pydantic para
cada una y no parsees texto con regex.

### 4.4 El ruteo *(Clase 7)*

Las bifurcaciones condicionales que muestra el diagrama, con la función **router declarada en el
grafo** y no un `if` adentro de un nodo. La que no podés saltear es la de `validar_pedido`: es la que
decide si el pedido se confirma, se escala o se rechaza.

### 4.5 Frenar, pedir aprobación y poder retomar *(Clase 7)*

`interrupt()` en el nodo de aprobación, con dos patrones: **aprobar y rechazar**. El payload tiene que
traer lo que la persona necesita para decidir sin abrir otra pantalla —razón social, crédito
disponible, neto, alertas y los renglones—: un interrupt que dice `{"aprobar": "sí/no"}` no sirve. Y
`SqliteSaver` sobre archivo, con un `thread_id` por mensaje, para poder **cortar el proceso a la mitad
de una aprobación, cerrar la terminal, volver a abrirla y retomar ese mismo pedido**.

### 4.6 Medir con los veinte casos

Escribí un script que corra los 20 mensajes contra el golden set y reportá dos números: **cuántos
salieron por la rama correcta**, y **en cuántas de las 8 consultas aparece el dato clave** (el monto
del flete, los días de garantía). Es muy común entregar un grafo que anda lindo con tres ejemplos
elegidos a mano y no tener idea de cómo se comporta con los otros diecisiete.

---

## 5. Por dónde empezar

El orden en que conviene construirlo, que no es el orden en que se lee el diagrama:

1. **El RAG primero.** Indexá los PDF, persistí el índice, y hacé que una pregunta suelta
   —*"cuánto sale el flete a Neuquén"*— se responda con la cita del documento y la página. Sin grafo
   ni nada: un script que anda.
2. **Las dos funciones que leen los CSV.** Primero como funciones de Python comunes, y recién cuando
   devuelven lo que esperás, ponéles el decorador y el `args_schema`.
3. **El estado y los nodos.** Definí el `TypedDict`, escribí cada nodo por separado y probalo con un
   diccionario armado a mano. Un nodo que anda solo es un nodo que después cablea fácil.
4. **El grafo.** Cableá los nodos, los routers y las aristas condicionales. Que compile y se dibuje.
5. **El frenado con aprobación.** El `interrupt()`, el checkpointer sobre archivo, y la vuelta con
   `Command(resume=...)`. Probá cortando el proceso de verdad.
6. **La medición, al final.** Los 20 casos y los dos números.

Una estructura que funciona, como sugerencia y no como obligación:

```
mesa_pedidos/     el paquete: indexado, retriever, tools, estado, nodos, grafo
datos/            los PDF, los CSV, la bandeja y el golden set, tal como te los damos
artefactos/       el índice persistido y los checkpoints (no se versionan)
correr.py         procesa un mensaje y maneja el pedido de aprobación
evaluar.py        corre los 20 casos y saca los dos números
```

---

## 6. Si querés ir más allá

Si te sobra tiempo y te interesa. Elegí uno y hacelo bien: cinco a medias valen menos que uno completo.

| Qué | Clase |
|---|---|
| **Retriever híbrido**: BM25 + denso fusionados en un ranking. Hay un caso esperándote —el MSG-004 pregunta por el SKU `HE-4330` y el catálogo tiene además el `HE-4335` y el `HE-4410`, que para un modelo de embeddings son casi el mismo punto—. Probá con densa sola, mirá qué trae, y después agregá BM25. | 5 |
| El tercer patrón del `interrupt()`: **editar** el pedido —menos unidades, otro descuento— y recalcular el total antes de confirmar | 7 |
| Una **segunda corrida** de la medición con otra configuración (densa sola contra híbrido) y la comparación de las dos | 5 |
| **Precisión y recall** de la extracción de items: de los que extrajiste, cuántos estaban en el mensaje, y al revés | — |
| Una tool `cotizar_flete` con zonas, bonificación por monto y recargos | 6 |
| `RetryPolicy` en los nodos que llaman al LLM, con fallback a un modelo más barato | 7 |
| Un subgrafo: encapsular la validación como grafo propio y usarlo como un nodo | 7 |
| Reranking con Cohere, o trazas en LangSmith, o un agente ReAct para la rama `derivar` | 5 / 3 / 6 |

---

