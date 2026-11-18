"""Prompt few-shot de Löffler et al. (2025), Sección 4.2, con su traducción al español (notas 6–8)."""
import numpy as np

# Instrucción inicial del prompt
CABECERA = ("Dados los siguientes ejemplos de cláusulas y sus etiquetas, prediga la última cláusula según su "
            "etiqueta. Responda sólo con las etiquetas correctas. No explique su decisión. A continuación, se "
            "indican todas las opciones posibles, seguidas de ejemplos típicos.")
UNA = "Una cláusula sólo puede tener una etiqueta."
VARIAS = "Una cláusula puede tener entre una y tres etiquetas."


# Ordena por longitud, divide en n grupos y toma la cláusula del medio de cada uno
def representativos(textos, n):
    """Muestreo representativo del paper: divide por longitud en n grupos y toma la mediana de cada uno."""
    textos = sorted(textos, key=len)
    if len(textos) <= n:
        return textos
    grupos = np.array_split(np.arange(len(textos)), n)
    return [textos[g[len(g) // 2]] for g in grupos if len(g)]


# Arma el prompt con la instrucción, las etiquetas posibles, los ejemplos y la cláusula nueva
def construir(clausula, ejemplos, etiquetas_posibles, multietiqueta):
    """ejemplos: lista de (texto, etiqueta_str)."""
    lineas = [CABECERA, VARIAS if multietiqueta else UNA, str(etiquetas_posibles), ""]
    lineas += [f"Cláusula: {t} Etiqueta: {e}" for t, e in ejemplos]
    lineas.append(f"Cláusula: {clausula} Etiqueta:")
    return "\n".join(lineas)
