"""Resume las corridas (media ± desviación entre semillas) y las compara con el paper."""
import json
from pathlib import Path

import pandas as pd

# Löffler et al. (2025), Tabla 5, Dark Detection (µ-F1, m-F1)
# [PASO 9] Valores del paper para comparar (Tabla 5, Dark Detection)
REFERENCIA = pd.DataFrame([
    {"metodo": "SVM + TF-IDF (paper)", "micro_f1": 0.96, "macro_f1": 0.79},
    {"metodo": "BETO ajustado (paper)", "micro_f1": 0.95, "macro_f1": 0.80},
    {"metodo": "GPT-4o-mini few-shot (paper)", "micro_f1": 0.86, "macro_f1": 0.66},
    {"metodo": "Llama 3.1 8B few-shot (paper)", "micro_f1": 0.57, "macro_f1": 0.46},
])


# [PASO 8] Promedio y desviación entre semillas → resultados/resumen.csv
def main():
    filas = []
    for f in Path("resultados").rglob("*.json"):
        fuente, tarea = f.parts[-3], f.parts[-2]
        for c in json.loads(f.read_text(encoding="utf-8")):
            filas.append({"fuente": fuente, "tarea": tarea, **c})
    if not filas:
        print("No hay corridas en resultados/")
        return
    df = pd.DataFrame(filas)
    df["n"] = df["n"].fillna(0).astype(int)
    res = (df.groupby(["fuente", "tarea", "metodo", "n"])
             .agg(micro_f1=("micro_f1", "mean"), micro_sd=("micro_f1", "std"),
                  macro_f1=("macro_f1", "mean"), macro_sd=("macro_f1", "std"),
                  segundos=("segundos", "mean"), corridas=("semilla", "count"))
             .round(3).reset_index())
    res.to_csv("resultados/resumen.csv", index=False)
    REFERENCIA.to_csv("resultados/referencia_paper.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))
    print("\nReferencia del paper (Tabla 5, Dark Detection):")
    print(REFERENCIA.to_string(index=False))


if __name__ == "__main__":
    main()
