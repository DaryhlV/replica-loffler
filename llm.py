"""Cliente de Ollama (servidor local) con reintentos; backend simulado para probar sin GPU."""
import re
import time

import requests

URL = "http://localhost:11434/api/chat"


# Envía el prompt al modelo de Ollama y devuelve su respuesta
class Ollama:
    def __init__(self, modelo, temperatura=0.0):
        self.modelo, self.temperatura = modelo, temperatura

    def __call__(self, prompt, semilla=0):
        cuerpo = {"model": self.modelo, "stream": False,
                  "messages": [{"role": "user", "content": prompt}],
                  "options": {"temperature": self.temperatura, "seed": semilla, "num_predict": 20, "num_ctx": 8192}}
        if self.modelo.startswith("qwen3"):
            cuerpo["think"] = False
        for intento in range(3):
            try:
                r = requests.post(URL, json=cuerpo, timeout=180)
                r.raise_for_status()
                return re.sub(r"<think>.*?</think>", "", r.json()["message"]["content"], flags=re.S).strip()
            except Exception as e:  # noqa: BLE001
                if intento == 2:
                    raise RuntimeError(f"Ollama no respondió: {e}") from e
                time.sleep(2 * (intento + 1))


class Simulado:
    """Respuestas falsas por palabras clave. Solo para probar el flujo; NO son resultados."""
    modelo = "simulado"

    def __call__(self, prompt, semilla=0):
        c = prompt.rsplit("Cláusula:", 1)[-1].lower()
        if "'abusiva'" in prompt.split("\n")[2]:
            return "abusiva" if any(k in c for k in ("sin previo", "discreción", "no será responsable", "renuncia")) else "no abusiva"
        return "ter" if any(k in c for k in ("cancelar", "cerrar", "bloquear", "terminado")) else "cr"


# Extrae las etiquetas válidas de la respuesta del modelo
def interpretar(texto, posibles):
    """Extrae las etiquetas válidas de la respuesta (las más largas primero: 'no abusiva' antes que 'abusiva')."""
    t = texto.lower()
    hallados = []
    for et in sorted(posibles, key=len, reverse=True):
        if re.search(r"(?<![\w])" + re.escape(et) + r"(?![\w])", t):
            hallados.append(et)
            t = t.replace(et, " ")
    return hallados
