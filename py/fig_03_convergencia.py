"""Figura: convergencia del estimador Monte Carlo según el método de muestreo."""

import numpy as np
import pandas as pd

import rutas
from estilo_figuras import CAT, TEXTO_2, guardar, plt

NOMBRES = {"mc": "Monte Carlo estándar", "antitetica": "Variables antitéticas", "sobol": "Sobol aleatorizado (QMC)"}


def main():
    df = pd.read_csv(rutas.SALIDAS / "convergencia.csv")
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for col, (metodo, g) in zip(CAT, df.groupby("metodo", sort=False)):
        ax.loglog(g["n"], g["rmse"], "o-", color=col, ms=5, label=NOMBRES[metodo])
        pend = np.polyfit(np.log(g["n"]), np.log(g["rmse"]), 1)[0]
        ax.annotate(f"pendiente {pend:.2f}", (g["n"].iloc[-1], g["rmse"].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", fontsize=7, color=TEXTO_2, va="center")
    mc = df[df["metodo"] == "mc"]
    n0, e0 = mc["n"].iloc[0], mc["rmse"].iloc[0]
    nn = np.array([mc["n"].min(), mc["n"].max()])
    ax.loglog(nn, e0 * (nn / n0) ** -0.5, color=TEXTO_2, lw=0.9, ls=":", label="Referencia n^(-1/2)")
    ax.set_xlabel("Número de escenarios n")
    ax.set_ylabel("RMSE de E[VP] (billones de COP)")
    ax.set_title("Convergencia del valor presente esperado del diseño base")
    ax.legend(loc="lower left")
    ax.set_xlim(nn[0] * 0.8, nn[1] * 2.2)
    guardar(fig, rutas.FIGURAS / "fig_convergencia.pdf")


if __name__ == "__main__":
    main()
