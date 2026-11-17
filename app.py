"""MVP: detector de cláusulas abusivas en español (SVM vs. LLM few-shot).

    streamlit run app.py
"""
import streamlit as st

import datos
import prompts
from categorias import CLASES, CLAVES, DETECCION
from experimento import correr_svm, ejemplos_fewshot, preparar
from llm import Ollama, Simulado, interpretar

# [PASO 10] MVP: interfaz web que compara SVM y LLM sobre una cláusula
st.set_page_config(page_title="Detector de cláusulas abusivas", page_icon="⚖️", layout="centered")
st.title("⚖️ Detector de cláusulas abusivas")
st.caption("Réplica de Löffler et al. (2025), Artificial Intelligence and Law · SVM vs. LLM few-shot con Ollama")


@st.cache_data
def base():
    return datos.cargar("sintetico")


df = base()
c1, c2 = st.columns(2)
modelo = c1.selectbox("Modelo LLM", ["llama3.1:8b", "qwen2.5:7b", "qwen3:8b", "ninguno (solo SVM)", "simulado"],
                      help="Sin Ollama en esta computadora, elige «ninguno (solo SVM)».")
n = c2.select_slider("Ejemplos por clase (n)", [1, 5, 10], value=5)
clausula = st.text_area("Cláusula", "El Banco podrá bloquear o cerrar la cuenta del Cliente en cualquier momento, sin previo aviso.", height=110)

if st.button("Analizar", type="primary") and clausula.strip():
    import pandas as pd
    det, _ = preparar(df, "deteccion")
    cla, _ = preparar(df, "clasificacion")
    prueba = pd.DataFrame({"texto": [clausula]})
    svm = correr_svm(det[det["split"] == "train"], prueba, DETECCION)[0][0]
    if modelo.startswith("ninguno"):
        st.metric("SVM + TF-IDF", svm)
        st.info("Modo sin LLM. Los resultados de Llama y Qwen están en resultados/resumen.csv y figuras/.")
        st.stop()
    llm = Simulado() if modelo == "simulado" else Ollama(modelo)
    with st.spinner("Consultando al LLM…"):
        try:
            r = llm(prompts.construir(clausula, ejemplos_fewshot(det[det["split"] != "test"], DETECCION, n, 0), DETECCION, False))
            llm_det = (interpretar(r, DETECCION) or ["sin respuesta válida"])[0]
            tipos = []
            if llm_det == "abusiva":
                r2 = llm(prompts.construir(clausula, ejemplos_fewshot(cla[cla["split"] != "test"], CLAVES, n, 0), CLAVES, True))
                tipos = interpretar(r2, CLAVES)
        except RuntimeError as e:
            st.error(f"{e}. ¿Está abierto Ollama y descargado el modelo?")
            st.stop()
    a, b = st.columns(2)
    a.metric("SVM + TF-IDF", svm)
    b.metric(f"{modelo} ({n}-shot)", llm_det)
    if tipos:
        st.warning("Tipo de cláusula según el LLM: " + "; ".join(f"**{CLASES[t][0]}** ({CLASES[t][1]})" for t in tipos))
    st.caption("Resultado orientativo; no constituye asesoría legal.")
