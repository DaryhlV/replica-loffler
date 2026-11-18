"""Carga de datos: sintético (español), CSV propio (p. ej. cláusulas bancarias peruanas) o LexGLUE (inglés).

Formato común: columnas texto, etiquetas (lista de claves; vacía = no abusiva) y split (train/val/test).
"""
import pandas as pd

from categorias import CLAVES


# Carga el corpus como una tabla con texto, etiquetas y split
def cargar(fuente="sintetico", csv=None):
    if fuente in ("sintetico", "csv"):
        ruta = csv or "data/sintetico.csv"
        df = pd.read_csv(ruta).fillna({"etiquetas": ""})
        df["etiquetas"] = df["etiquetas"].map(lambda s: [c for c in str(s).split(";") if c])
        if "split" not in df.columns:
            raise ValueError("El CSV necesita una columna split con train, val o test")
        return df[["texto", "etiquetas", "split"]]
    if fuente == "lexglue":  # carga UNFAIR-ToS (inglés) desde Hugging Face
        mapa = {0: "ltd", 1: "ter", 2: "cr"}  # las demás categorías quedan como abusivas genéricas
        from datasets import load_dataset
        partes = []
        for split in ("train", "validation", "test"):
            ds = load_dataset("coastalcph/lex_glue", "unfair_tos", split=split)
            partes.append(pd.DataFrame({
                "texto": ds["text"],
                "etiquetas": [[mapa.get(i, "otra") for i in ls] for ls in ds["labels"]],
                "split": "val" if split == "validation" else split}))
        return pd.concat(partes, ignore_index=True)
    raise ValueError(f"Fuente desconocida: {fuente}")


def deteccion(etiquetas):
    return "abusiva" if etiquetas else "no abusiva"


def solo_clases(etiquetas):
    return [e for e in etiquetas if e in CLAVES]
