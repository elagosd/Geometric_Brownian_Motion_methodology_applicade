"""Figuras de sensibilidad: tornado y tasa de descuento."""

import json

import numpy as np
import pandas as pd

import rutas
from estilo_figuras import CAT, TEXTO, TEXTO_2, guardar, plt


def fmt(v):
    if isinstance(v, (bool, np.bool_)) or str(v) in ("True", "False"):
        return "sí" if str(v) == "True" else "no"
    v = float(v)
    if v < 1:
        return f"{v:.3g}".replace(".", ",")
    return f"{v:,.0f}".replace(",", ".")


def tornado():
    df = pd.read_csv(rutas.SALIDAS / "sensibilidad.csv")
    c = json.loads((rutas.SALIDAS / "sensibilidad_central.json").read_text(encoding="utf-8"))
    central = c["bn_central"]
    df = df.sort_values("rango")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(df))
    for yi, (_, f) in zip(y, df.iterrows()):
        ax.plot([f["bn_bajo"], f["bn_alto"]], [yi, yi], color=TEXTO_2, lw=1.0, zorder=1)
        ax.scatter(f["bn_bajo"], yi, color=CAT[0], s=28, zorder=2, label="Valor bajo" if yi == 0 else None)
        ax.scatter(f["bn_alto"], yi, color=CAT[1], s=28, zorder=2, label="Valor alto" if yi == 0 else None)
        ax.annotate(fmt(f["bajo"]), (f["bn_bajo"], yi), xytext=(0, 5), textcoords="offset points",
                    fontsize=6.5, ha="center", color=TEXTO_2)
        ax.annotate(fmt(f["alto"]), (f["bn_alto"], yi), xytext=(0, 5), textcoords="offset points",
                    fontsize=6.5, ha="center", color=TEXTO_2)
    ax.axvline(central, color=TEXTO, lw=0.8, ls="--")
    ax.axvline(0, color=TEXTO, lw=0.6)
    ax.set_yticks(y, df["factor"], fontsize=7.5)
    ax.set_xlabel(f"Beneficio neto esperado de «{c['diseno']}» (billones de COP)")
    ax.set_title("Sensibilidad un factor a la vez (línea discontinua: caso central)")
    ax.legend(loc="lower right")
    guardar(fig, rutas.FIGURAS / "fig_tornado.pdf")


def descuento():
    df = pd.read_csv(rutas.SALIDAS / "descuento.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for col, (nom, g) in zip(CAT, df.groupby("diseno", sort=False)):
        ax.plot(g["tasa"] * 100, g["bn_media"], "o-", color=col, ms=4, label=nom)
    ax.axhline(0, color=TEXTO, lw=0.8)
    ax.axvline(9, color=TEXTO_2, lw=0.8, ls=":")
    ax.text(9.1, ax.get_ylim()[1], "DNP: 9 %", fontsize=7, color=TEXTO_2, va="top")
    ax.set_xlabel("Tasa social de descuento (%)")
    ax.set_ylabel("Beneficio neto esperado (billones de COP)")
    ax.set_title("El signo de las líneas de metro depende de la tasa de descuento")
    ax.legend(loc="upper right", fontsize=7)
    guardar(fig, rutas.FIGURAS / "fig_descuento.pdf")


if __name__ == "__main__":
    tornado()
    descuento()
