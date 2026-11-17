# Guía para la demo y las preguntas

## Dónde está cada paso del workflow en el código

Busca `[PASO` en VS Code (Ctrl + Shift + F) para saltar a cada parte.

| Paso del workflow | Archivo | Qué mirar |
|---|---|---|
| 1-2 · Términos y condiciones → corpus en español | `generar_sinteticos.py` | listas `ABUSIVAS` y `JUSTAS`, función `rellenar` |
| 2 · Carga del corpus | `datos.py` | función `cargar` (sintético, CSV propio o LexGLUE) |
| 3 · Anotación: 6 cláusulas oscuras | `categorias.py` | diccionario `CLASES` (cr, ter, ch, er, ltd, nod) |
| 4 · Partición train / val / test | `generar_sinteticos.py` | función `reservar` y bucle `for parte, total in ...` |
| 5 · Supervisado: TF-IDF + SVM | `experimento.py` | función `correr_svm` |
| 6 · Prompt few-shot | `prompts.py` | `CABECERA`, `representativos`, `construir` |
| 6 · Elegir n ejemplos por clase | `experimento.py` | función `ejemplos_fewshot` |
| 7 · LLM con Ollama | `llm.py` | clase `Ollama` (temperatura 0, semilla) e `interpretar` |
| 7 · Bucle few-shot (n y semillas) | `experimento.py` | `for n in a.n:` dentro de `main` |
| 8 · Métricas µ-F1 y macro-F1 | `experimento.py`, `evaluar.py` | función `metricas`; promedio en `evaluar.py` |
| 9 · Comparar con el paper | `evaluar.py`, `graficos.py` | `REFERENCIA` (Tabla 5) y el gráfico |
| 10 · MVP | `app.py` | interfaz Streamlit |

## Comandos de la demo (en este orden)

```
python generar_sinteticos.py
python experimento.py --metodo svm --tarea deteccion
python experimento.py --metodo llm --modelo llama3.1:8b --tarea deteccion --n 5 --semillas 1 --limite 10
python evaluar.py
streamlit run app.py
```

El tercer comando usa solo 10 cláusulas para que termine en segundos durante la demo. Los resultados
completos ya están en `resultados/resumen.csv` y `figuras/`.

## Preguntas probables

- **¿Por qué datos sintéticos?** El corpus chileno del paper no es público; el profesor permitió datos
  sintéticos. Mantengo la proporción (20 % abusivas) y las 6 clases oscuras del paper.
- **¿Cómo evitas que el SVM memorice?** La mitad del test usa plantillas que no aparecen en
  entrenamiento (`reservar` en `generar_sinteticos.py`).
- **¿Qué es TF-IDF?** Convierte cada cláusula en un vector según qué tan importantes son sus palabras;
  el SVM aprende una frontera entre abusivas y no abusivas.
- **¿Qué es few-shot?** No se entrena el LLM: se le muestran n ejemplos etiquetados dentro del prompt y
  se le pide la etiqueta de la cláusula nueva.
- **¿Por qué temperatura 0?** Para que el modelo responda siempre lo mismo y el experimento sea
  replicable.
- **¿Por qué macro-F1?** Las clases están desbalanceadas (80 % / 20 %); macro-F1 da el mismo peso a la
  clase minoritaria (abusiva).
- **¿Qué pasa si el LLM responde algo raro?** `interpretar` busca solo etiquetas válidas; si no hay
  ninguna, se cuenta como respuesta inválida y se asigna «no abusiva».
- **¿Qué falta respecto al paper?** El ajuste fino de transformers (BETO, XLM-R). En la tesis lo haré con
  QLoRA sobre Llama 3.1 8B.
