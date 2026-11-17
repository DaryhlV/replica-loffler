"""Gráfico principal: macro-F1 según número de ejemplos (n) para cada LLM, frente al SVM y al paper."""
import matplotlib.pyplot as plt
import pandas as pd

AZUL, NARANJA, VERDE, TINTA2, REJILLA = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA2})


# [PASO 9] Gráfico: macro-F1 según n, frente al SVM y al paper
def main(fuente="sintetico"):
    res = pd.read_csv("resultados/resumen.csv")
    ref = pd.read_csv("resultados/referencia_paper.csv")
    for tarea in res["tarea"].unique():
        d = res[(res["fuente"] == fuente) & (res["tarea"] == tarea)]
        if d.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        colores = iter([AZUL, NARANJA, VERDE, "#6250d6"])
        for m, g in d[d["n"] > 0].groupby("metodo"):
            g = g.sort_values("n")
            c = next(colores)
            ax.errorbar(g["n"], g["macro_f1"], yerr=g["macro_sd"].fillna(0), color=c, marker="o", ms=8, lw=2.5,
                        capsize=4, label=f"{m} few-shot (réplica)")
            ax.annotate(f"{g['macro_f1'].iloc[-1]:.2f}", (g["n"].iloc[-1], g["macro_f1"].iloc[-1]),
                        xytext=(8, 0), textcoords="offset points", va="center", fontsize=10)
        svm = d[d["metodo"].str.startswith("SVM")]
        if len(svm):
            v = svm["macro_f1"].iloc[0]
            ax.axhline(v, color="#14213d", lw=2, ls="--", label=f"SVM + TF-IDF (réplica) = {v:.2f}")
        if tarea == "deteccion":
            for _, r in ref.iterrows():
                if r["metodo"].startswith(("SVM", "Llama")):
                    ax.axhline(r["macro_f1"], color="#9aa0b4", lw=1.2, ls=":")
                    ax.text(10.4, r["macro_f1"], r["metodo"], fontsize=8.5, color=TINTA2, va="center",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
        ax.set_xticks([1, 5, 10])
        ax.set_xlim(0.5, 13.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Ejemplos por clase en el prompt (n)")
        ax.set_ylabel("Macro-F1")
        titulo = "Detección de cláusulas abusivas" if tarea == "deteccion" else "Clasificación de cláusulas oscuras"
        ax.set_title(f"{titulo}: few-shot vs. SVM", loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=9, loc="lower left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.yaxis.grid(True, color=REJILLA)
        ax.set_axisbelow(True)
        if fuente == "sintetico":
            fig.text(0.99, 0.01, "Datos sintéticos en español", ha="right", fontsize=8, color="#b91c1c")
        fig.tight_layout()
        fig.savefig(f"figuras/{fuente}_{tarea}.png", dpi=300)
        plt.close(fig)
    print("Gráficos en figuras/")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "sintetico")
