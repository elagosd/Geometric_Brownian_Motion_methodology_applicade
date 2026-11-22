"""Figuras de diseños: frontera media-CVaR, distribuciones y componentes del costo."""

import json

import numpy as np
import pandas as pd

import rutas
from estilo_figuras import CAT, NEUTRO, SEQ, TEXTO, TEXTO_2, guardar, plt

BASE = "Base | θ=0"


def frontera():
    df = pd.read_csv(rutas.SALIDAS / "resumen_disenos.csv")
    corrida = json.loads((rutas.SALIDAS / "corrida.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    marcas = {0.0: "o", 0.5: "s", 1.0: "D"}
    for th, g in df.groupby("theta"):
        ax.scatter(g["vp_media"], g["vp_cvar"], marker=marcas[th], s=26, color="white",
                   edgecolor=NEUTRO, lw=1.0, label=f"θ = {th:g}", zorder=2)
    par = df[df["pareto"]].sort_values("vp_media")
    ax.plot(par["vp_media"], par["vp_cvar"], "-", color=CAT[0], lw=1.6, zorder=3, label="Frontera de Pareto")
    ax.scatter(par["vp_media"], par["vp_cvar"], s=30, color=CAT[0], zorder=4)
    destacar = {BASE: "Base", corrida["optimo_neutral"]: "Óptimo", "L1X+L2+L3+RTN | θ=1": "Todo"}
    for nom, et in destacar.items():
        f = df[df["diseno"] == nom].iloc[0]
        ax.annotate(f"{et}: {nom}", (f["vp_media"], f["vp_cvar"]), xytext=(8, -10 if et != "Todo" else 6),
                    textcoords="offset points", fontsize=7, color=TEXTO,
                    arrowprops=dict(arrowstyle="-", color=TEXTO_2, lw=0.6))
    ax.set_xlabel("Costo social esperado E[VP] (billones de COP de 2026)")
    ax.set_ylabel("CVaR 95 % del VP (billones de COP)")
    ax.set_title("Riesgo y costo de los 48 diseños (menor es mejor en ambos ejes)")
    ax.legend(loc="lower right")
    guardar(fig, rutas.FIGURAS / "fig_frontera.pdf")


def distribuciones():
    datos = np.load(rutas.SALIDAS / "vp_disenos.npz", allow_pickle=True)
    nombres = list(datos["nombres"])
    vp = datos["vp"]
    corrida = json.loads((rutas.SALIDAS / "corrida.json").read_text(encoding="utf-8"))
    base = vp[nombres.index(BASE)]
    sel = [corrida["optimo_neutral"], "Base | θ=1", "RTN | θ=0", "L3 | θ=0", "L1X+L2+L3+RTN | θ=1"]
    sel = list(dict.fromkeys([s for s in sel if s in nombres]))
    fig, axes = plt.subplots(len(sel), 1, figsize=(6.4, 1.15 * len(sel) + 0.6), sharex=True)
    todos = np.concatenate([base - vp[nombres.index(s)] for s in sel])
    bins = np.linspace(np.percentile(todos, 0.5), np.percentile(todos, 99.5), 60)
    for ax, s, col in zip(axes, sel, CAT):
        bn = base - vp[nombres.index(s)]
        ax.hist(bn, bins=bins, color=col, alpha=0.85, edgecolor="white", lw=0.4)
        ax.axvline(0, color=TEXTO, lw=0.8)
        ax.axvline(bn.mean(), color=TEXTO_2, lw=0.8, ls="--")
        ax.set_yticks([])
        ax.grid(False, axis="y")
        ax.text(0.01, 0.92, f"{s}   media {bn.mean():+.1f}   P(>0) = {np.mean(bn > 0):.0%}",
                transform=ax.transAxes, fontsize=7.5, va="top", color=TEXTO)
    axes[-1].set_xlabel("Beneficio neto frente al diseño base (billones de COP, VP 2026)")
    axes[0].set_title("Distribución del beneficio neto por escenario (línea discontinua: media)")
    guardar(fig, rutas.FIGURAS / "fig_distribucion.pdf")


def componentes():
    tray = json.loads((rutas.SALIDAS / "trayectorias.json").read_text(encoding="utf-8"))
    base = tray[BASE]["componentes_vp"]
    nombres = [n for n in tray if n != BASE]
    etiquetas = {"tiempo": "Tiempo", "auto": "Auto (operación y externalidades)", "bus": "Operación de buses",
                 "om": "O&M férrea", "suelo": "Política de suelo", "capex": "CAPEX", "residual": "Valor residual"}
    comps = list(etiquetas)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    y = np.arange(len(nombres))
    for k, (comp, col) in enumerate(zip(comps, CAT)):
        delta = np.array([tray[n]["componentes_vp"][comp] - base[comp] for n in nombres])
        ax.barh(y + (k - 3) * 0.11, delta, height=0.1, color=col, label=etiquetas[comp], zorder=3)
    neto = np.array([sum(base.values()) - sum(tray[n]["componentes_vp"].values()) for n in nombres])
    for yi, v in zip(y, neto):
        ax.text(1.01, yi, f"BN {v:+.1f}", transform=ax.get_yaxis_transform(), fontsize=7, va="center", color=TEXTO)
    ax.axvline(0, color=TEXTO, lw=0.8)
    ax.set_yticks(y, nombres, fontsize=7.5)
    ax.set_xlabel("Cambio en el VP de cada componente frente a la base (billones de COP)")
    ax.set_title("De dónde salen los costos y ahorros de cada diseño")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3, fontsize=7)
    guardar(fig, rutas.FIGURAS / "fig_componentes.pdf")


if __name__ == "__main__":
    frontera()
    distribuciones()
    componentes()
