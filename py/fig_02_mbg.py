"""Figuras del MBG: calibración con TransMilenio, abanico de demanda y zonas."""

import json

import numpy as np

import rutas
from estilo_figuras import CAT, SEQ, TEXTO_2, guardar, plt
from modelo import gbm
from modelo.parametros import parametros_calibrados


def calibracion():
    d = json.loads((rutas.SALIDAS / "calibracion_mbg.json").read_text(encoding="utf-8"))
    anios = np.array(d["anios"])
    vals = np.array(d["valores"])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(anios, vals, "o-", color=CAT[0], ms=5, label="Observado (pasajeros/día, millones)")
    t = np.linspace(0, 17, 200)
    for clave, col, nombre in (("completo", CAT[1], "MBG con todos los datos"),
                               ("sin_pandemia", CAT[2], "MBG sin choque ni rebote 2020–2023")):
        a = d[clave]
        q = np.array([gbm.cuantiles_teoricos(a["mu"], a["sigma"], ti, np.array([0.05, 0.5, 0.95]))
                      for ti in t]) * vals[0]
        ax.plot(2008 + t, q[:, 1], color=col, lw=1.4, label=f"{nombre}: mediana (σ = {a['sigma']:.1%})")
        ax.fill_between(2008 + t, q[:, 0], q[:, 2], color=col, alpha=0.12, lw=0)
    ax.set_ylim(0, 6)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel("Año")
    ax.set_ylabel("Millones de pasajeros/día")
    ax.set_title("Calibración del MBG con la demanda troncal de TransMilenio (banda 5–95 %)")
    ax.legend(loc="upper left")
    guardar(fig, rutas.FIGURAS / "fig_mbg_calibracion.pdf")


def abanico():
    p = parametros_calibrados()
    e = np.load(rutas.SALIDAS / "escenarios.npz")
    anios, tot = e["anios"], e["total"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bandas = [(5, 95), (25, 75)]
    for (lo, hi), col in zip(bandas, (SEQ[1], SEQ[3])):
        ax.fill_between(anios, np.percentile(tot, lo, 0), np.percentile(tot, hi, 0), color=col,
                        alpha=0.6, lw=0, label=f"Percentiles {lo}–{hi}")
    for k in range(8):
        ax.plot(anios, tot[k], color=TEXTO_2, lw=0.5, alpha=0.6)
    ax.plot(anios, np.median(tot, 0), color=SEQ[6], lw=1.8, label="Mediana simulada")
    ax.plot(anios, tot.mean(0), color=CAT[1], lw=1.4, ls="--", label="Media simulada")
    ax.set_xlabel("Año")
    ax.set_ylabel("Índice de demanda motorizada (2025 = 1)")
    ax.set_title(f"Demanda agregada simulada: {tot.shape[0]} trayectorias del MBG multivariado")
    ax.legend(loc="upper left")
    guardar(fig, rutas.FIGURAS / "fig_mbg_abanico.pdf")

    Xf = e["X_final"]
    nombres = p.zonas["codigo"].tolist()
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    pos = np.arange(len(nombres))
    q = np.percentile(Xf, [5, 25, 50, 75, 95], axis=0)
    ax.vlines(pos, q[0], q[4], color=SEQ[3], lw=1.2)
    ax.vlines(pos, q[1], q[3], color=SEQ[5], lw=5)
    ax.scatter(pos, q[2], color="white", edgecolor=SEQ[6], s=18, zorder=3, label="Mediana")
    ax.axhline(1.0, color=TEXTO_2, lw=0.8)
    ax.set_xticks(pos, nombres)
    ax.set_ylabel("Índice en 2050 (2025 = 1)")
    ax.set_title("Distribución del índice de demanda por zona en 2050 (barra: 25–75 %; línea: 5–95 %)")
    ax.legend(loc="upper left")
    guardar(fig, rutas.FIGURAS / "fig_mbg_zonas.pdf")


if __name__ == "__main__":
    calibracion()
    abanico()
