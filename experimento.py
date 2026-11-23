"""Réplica de Löffler et al. (2025): SVM + TF-IDF (supervisado) vs. LLM few-shot (Ollama).

Tareas:
  deteccion      ¿la cláusula es abusiva? (como "Dark Detection" del paper)
  clasificacion  ¿qué tipo de cláusula oscura es? (multietiqueta, como "Dark Classification")

Ejemplos:
  python experimento.py --metodo svm --tarea deteccion
  python experimento.py --metodo llm --modelo llama3.1:8b --tarea deteccion --n 1 5 10 --semillas 3
  python experimento.py --metodo llm --modelo qwen2.5:7b  --tarea clasificacion --n 5
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from tqdm import tqdm

import datos
import prompts
from categorias import CLAVES, DETECCION
from llm import Ollama, Simulado, interpretar


def preparar(df, tarea):
    df = df.copy()
    if tarea == "deteccion":
        df["y"] = df["etiquetas"].map(lambda e: [datos.deteccion(e)])
        posibles = DETECCION
    else:  # deja solo las cláusulas abusivas con su tipo
        df["y"] = df["etiquetas"].map(datos.solo_clases)
        df = df[df["y"].map(len) > 0]
        posibles = CLAVES
    return df, posibles


# Calcula micro-F1, macro-F1, precisión y recall macro
def metricas(y_true, y_pred, posibles):
    mlb = MultiLabelBinarizer(classes=posibles)
    Y, P = mlb.fit_transform(y_true), mlb.transform(y_pred)
    kw = dict(zero_division=0)
    return {"micro_f1": f1_score(Y, P, average="micro", **kw), "macro_f1": f1_score(Y, P, average="macro", **kw),
            "precision_macro": precision_score(Y, P, average="macro", **kw),
            "recall_macro": recall_score(Y, P, average="macro", **kw)}


# Entrena un SVM lineal con TF-IDF y predice las etiquetas de las cláusulas de prueba
def correr_svm(tr, te, posibles):
    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=1, sublinear_tf=True)
    X, Xt = vec.fit_transform(tr["texto"]), vec.transform(te["texto"])
    mlb = MultiLabelBinarizer(classes=posibles)
    clf = OneVsRestClassifier(LinearSVC(C=1.0, class_weight="balanced"))
    clf.fit(X, mlb.fit_transform(tr["y"]))
    pred = mlb.inverse_transform(clf.predict(Xt))
    if posibles == DETECCION:  # si no hay una sola etiqueta, toma la de mayor puntaje
        punt = clf.decision_function(Xt)
        pred = [[posibles[int(np.argmax(s))]] if len(p) != 1 else list(p) for p, s in zip(pred, punt)]
    return [list(p) for p in pred]


# Elige n ejemplos representativos por clase para el prompt
def ejemplos_fewshot(tr, posibles, n, semilla):
    """n ejemplos representativos por clase (el paper exige al menos n por clase)."""
    rnd = random.Random(semilla)
    ej = []
    for et in posibles:
        textos = tr[tr["y"].map(lambda y: et in y)]["texto"].tolist()
        rnd.shuffle(textos)
        for t in prompts.representativos(textos[: max(n * 4, n)], n):
            ej.append((t, ", ".join(tr.loc[tr["texto"] == t, "y"].iloc[0])))
    rnd.shuffle(ej)
    return ej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metodo", choices=["svm", "llm"], required=True)
    ap.add_argument("--modelo", default="llama3.1:8b", help="modelo de Ollama o 'simulado'")
    ap.add_argument("--tarea", choices=["deteccion", "clasificacion"], default="deteccion")
    ap.add_argument("--n", type=int, nargs="+", default=[1, 5, 10], help="ejemplos por clase")
    ap.add_argument("--semillas", type=int, default=3)
    ap.add_argument("--fuente", choices=["sintetico", "csv", "lexglue"], default="sintetico")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--limite", type=int, default=0, help="usar solo N cláusulas de prueba")
    a = ap.parse_args()

    df, posibles = preparar(datos.cargar(a.fuente, a.csv), a.tarea)
    tr = df[df["split"].isin(["train", "val"])]  # train + val para los ejemplos del prompt
    te = df[df["split"] == "test"]
    if a.limite:
        te = te.sample(n=min(a.limite, len(te)), random_state=0)
    salida = Path("resultados") / a.fuente / a.tarea
    salida.mkdir(parents=True, exist_ok=True)

    corridas = []
    if a.metodo == "svm":
        t0 = time.time()
        pred = correr_svm(df[df["split"] == "train"], te, posibles)
        corridas.append({"metodo": "SVM + TF-IDF", "n": None, "semilla": 0, "segundos": round(time.time() - t0, 1),
                         **metricas(te["y"].tolist(), pred, posibles)})
        nombre = "svm"
    else:
        llm = Simulado() if a.modelo == "simulado" else Ollama(a.modelo)
        # Consulta al LLM por cada cláusula de prueba, para cada n y cada semilla
        for n in a.n:
            for s in range(a.semillas):
                ej = ejemplos_fewshot(tr, posibles, n, s)
                t0, pred, crudas = time.time(), [], []
                for txt in tqdm(te["texto"], desc=f"{a.modelo} n={n} semilla={s}"):
                    r = llm(prompts.construir(txt, ej, posibles, a.tarea == "clasificacion"), semilla=s)
                    et = interpretar(r, posibles)
                    if a.tarea == "deteccion":
                        et = et[:1] or ["no abusiva"]  # respuesta inválida → no abusiva
                    pred.append(et[:3])
                    crudas.append(r)
                corridas.append({"metodo": a.modelo, "n": n, "semilla": s, "segundos": round(time.time() - t0, 1),
                                 "respuestas_invalidas": sum(not interpretar(r, posibles) for r in crudas),
                                 **metricas(te["y"].tolist(), pred, posibles)})
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in corridas[-1].items()})
        nombre = f"llm_{a.modelo.replace(':', '-')}"
    (salida / f"{nombre}.json").write_text(json.dumps(corridas, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado en {salida / (nombre + '.json')}  (prueba: {len(te)} cláusulas)")


if __name__ == "__main__":
    main()
