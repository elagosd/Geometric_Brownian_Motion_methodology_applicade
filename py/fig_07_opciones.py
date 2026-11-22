"""Figura: opción real de construir la Línea 3."""

import json

import numpy as np

import rutas
from estilo_figuras import CAT, SEQ, TEXTO, TEXTO_2, guardar, plt


def main():
    d = json.loads((rutas.SALIDAS / "opciones_reales.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    ax = axes[0]
    anios = np.array(d["anios_decision"])
    ax.plot(anios, d["vpn_por_anio"], "o-", color=CAT[0], ms=4, label="Comprometerse a decidir en ese año")
    ax.axhline(d["valor_flexible"], color=CAT[1], lw=1.6, label=f"Regla flexible (LSM): {d['valor_flexible']:.2f}")
    ax.axhline(0, color=TEXTO, lw=0.7)
    ax.set_xlabel("Año de la decisión de construir")
    ax.set_ylabel("VPN esperado (billones de COP)")
    ax.set_title("(a) Valor de invertir en la Línea 3", fontsize=9)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(loc="lower right", fontsize=7)

    ax = axes[1]
    a = np.array(d["beneficio_anios"])
    b = d["beneficio"]
    ax.fill_between(a, b["p5"], b["p95"], color=SEQ[1], lw=0, label="Beneficio potencial 5–95 %")
    ax.plot(a, b["p50"], color=SEQ[6], lw=1.6, label="Mediana")
    fr = np.array([np.nan if x is None else x for x in d["frontera"]], float)
    ok = ~np.isnan(fr)
    if ok.any():
        ax.plot(anios[ok], fr[ok], "s", color=CAT[1], ms=4, label="Frontera de ejercicio (LSM)")
    ax.set_xlabel("Año")
    ax.set_ylabel("Billones de COP por año")
    ax.set_title("(b) Ahorro operativo anual con la L3", fontsize=9)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    guardar(fig, rutas.FIGURAS / "fig_opciones.pdf")


if __name__ == "__main__":
    main()
