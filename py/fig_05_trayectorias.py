"""Figura: trayectorias de tiempo de viaje y velocidad vial con bandas de incertidumbre."""

import json

import numpy as np

import rutas
from estilo_figuras import CAT, guardar, plt

BASE = "Base | θ=0"


def main():
    tray = json.loads((rutas.SALIDAS / "trayectorias.json").read_text(encoding="utf-8"))
    corrida = json.loads((rutas.SALIDAS / "corrida.json").read_text(encoding="utf-8"))
    sel = list(dict.fromkeys([BASE, corrida["optimo_neutral"], "L1X+L2+L3+RTN | θ=1"]))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), layout="constrained")
    paneles = (("tiempo", "Tiempo medio de viaje (min)"), ("vel", "Velocidad vial pico (km/h)"),
               ("hac", "Carga/capacidad máx. del TP"))
    for ax, (clave, titulo) in zip(axes, paneles):
        for nom, col in zip(sel, CAT):
            t = tray[nom]
            a = np.array(t["anios"])
            ax.fill_between(a, t[clave]["p5"], t[clave]["p95"], color=col, alpha=0.15, lw=0)
            ax.plot(a, t[clave]["p50"], color=col, lw=1.6, label=nom)
        ax.set_title(titulo, fontsize=8.5)
        ax.set_xlabel("Año")
        ax.set_xticks([2030, 2040, 2050])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=7)
    fig.suptitle("Mediana y banda 5–95 % entre escenarios", fontsize=9.5)
    guardar(fig, rutas.FIGURAS / "fig_trayectorias.pdf")


if __name__ == "__main__":
    main()
