"""Figuras de la fase I (2013) y de la comparación elevado vs. subterráneo."""

import json

import numpy as np
import pandas as pd

import rutas
from estilo_figuras import CAT, NEUTRO, SEQ, TEXTO, TEXTO_2, guardar, plt
from modelo import metro as M

S = rutas.SALIDAS
# Rango del costo urbano anual del viaducto por pérdida de valor de los predios
# en una franja de 100 m a lado y lado, anualizado al 9-12 % (ver el capítulo del metro).
EXT_BAJA, EXT_ALTA = 0.06, 0.28


def abanico_2013():
    e = np.load(S / "fase2013_escenarios.npz")
    anios, tot, obs = e["anios"], e["total"], e["observado"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for (lo, hi), col in zip([(5, 95), (25, 75)], (SEQ[1], SEQ[3])):
        ax.fill_between(anios, np.percentile(tot, lo, 0), np.percentile(tot, hi, 0), color=col, alpha=0.6, lw=0,
                        label=f"Escenarios vistos desde 2013: percentiles {lo}–{hi}")
    ax.plot(anios, np.median(tot, 0), color=SEQ[6], lw=1.6, label="Mediana vista desde 2013")
    ax.plot(np.arange(2013, 2013 + obs.size), obs, "o-", color=CAT[1], ms=4, lw=1.6,
            label="Trayectoria observada (estilizada)")
    ax.set_xlim(2013, 2035)
    ax.set_ylim(0.5, 2.0)
    ax.set_xlabel("Año")
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.set_ylabel("Índice de demanda motorizada (2013 = 1)")
    ax.set_title("Lo que se esperaba en 2013 y lo que ocurrió")
    ax.legend(loc="upper left", fontsize=7)
    guardar(fig, rutas.FIGURAS / "fig_fase2013_abanico.pdf")


def beneficios_tasa():
    df = pd.read_csv(S / "fase2013_descuento.csv")
    df = df[df["diseno"].str.endswith("θ=0") & ~df["diseno"].str.startswith("Sin metro")]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    nombres = list(dict.fromkeys(df["diseno"]))
    y = np.arange(len(nombres))
    for (tasa, g), col, mk in zip(df.groupby("tasa"), CAT, ("o", "s", "D")):
        g = g.set_index("diseno").loc[nombres]
        ax.scatter(g["bn_media"], y, color=col, marker=mk, s=34, label=f"Tasa {100 * tasa:.1f} %".replace(".", ","), zorder=3)
    ax.axvline(0, color=TEXTO, lw=0.8)
    ax.set_yticks(y, [n.replace(" | θ=0", "").replace(", apertura", ",\napertura") for n in nombres], fontsize=8)
    ax.set_xlabel("Beneficio neto esperado frente a no hacer metro (billones de 2025, VP 2014)")
    ax.set_title("Evaluación ex ante (decisión en 2014), sin política de suelo")
    ax.legend(loc="lower left", fontsize=7)
    guardar(fig, rutas.FIGURAS / "fig_fase2013_beneficios.pdf")


def mapa_decision():
    """Costo urbano anual del viaducto que iguala ambas tipologías según el CAPEX subterráneo."""
    um = pd.read_csv(S / "fase2013_umbral.csv")
    capex_of = M.capex_total_2025("subterraneo_2015")
    c27 = M.comparar(27.0, 28)
    capex_alto = c27["capex_elevado"] + c27["ahorro_elevado"]
    capex_e = M.capex_total_2025("elevado_factibilidad_2017")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    xs = np.linspace(capex_e, capex_alto + 4, 100)
    for (tasa, g), col in zip(um.groupby("tasa"), CAT):
        g = g.set_index("alternativa")
        y_of = g.loc["Subterráneo (costo oficial) | θ=0", "externalidad_umbral"]
        y_al = g.loc["Subterráneo (costo unitarios) | θ=0", "externalidad_umbral"]
        pend = (y_al - y_of) / (capex_alto - capex_of)
        ax.plot(xs, y_of + pend * (xs - capex_of), color=col, lw=1.6,
                label=f"Tasa {100 * tasa:.1f} %".replace(".", ","))
        ax.scatter([capex_of, capex_alto], [y_of, y_al], color=col, s=22, zorder=3)
    ax.axhspan(EXT_BAJA, EXT_ALTA, color=NEUTRO, alpha=0.25, lw=0)
    ax.text(xs[-1], EXT_ALTA + 0.08, "Costo urbano estimado por pérdida\nde valor de los predios", ha="right",
            va="bottom", fontsize=7, color=TEXTO_2)
    ax.axhline(0, color=TEXTO, lw=0.7)
    for x, et in ((capex_of, "Costo oficial\n(diseño 2014–2015)"), (capex_alto, "Costos unitarios\nEMB 2022")):
        ax.axvline(x, color=TEXTO_2, lw=0.6, ls=":")
        ax.text(x, ax.get_ylim()[1] if False else 5.6, et, ha="center", va="top", fontsize=7, color=TEXTO_2)
    ax.set_ylim(-1, 6)
    ax.set_xlabel("CAPEX del metro subterráneo a la calle 127 (billones de 2025)")
    ax.set_ylabel("Costo urbano anual del viaducto\nque iguala ambas opciones (billones/año)")
    ax.set_title("Por encima de cada línea conviene el subterráneo; por debajo, el elevado")
    ax.legend(loc="center left", fontsize=7)
    guardar(fig, rutas.FIGURAS / "fig_metro_decision.pdf")


def tipologia_fase2():
    df = pd.read_csv(S / "fase2_tipologia.csv").set_index("diseno")
    pares = [("L2 | θ=0", "L2E | θ=0", "Línea 2"), ("L3 | θ=0", "L3E | θ=0", "Línea 3"),
             ("L3+RTN | θ=1", "L3E+RTN | θ=1", "L3 + Regiotram Norte, θ=1")]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(pares))
    for k, (sub, ele, et) in enumerate(pares):
        ax.plot([df.loc[sub, "bn_media"], df.loc[ele, "bn_media"]], [k, k], color=NEUTRO, lw=1.0, zorder=1)
        ax.scatter(df.loc[sub, "bn_media"], k, color=CAT[0], s=40, zorder=2, label="Subterránea" if k == 0 else None)
        ax.scatter(df.loc[ele, "bn_media"], k, color=CAT[1], s=40, marker="s", zorder=2,
                   label="Variante elevada" if k == 0 else None)
        ax.text(1.01, k, f"umbral {df.loc[ele, 'externalidad_umbral']:.1f} bill./año".replace(".", ","),
                transform=ax.get_yaxis_transform(), fontsize=7, va="center", color=TEXTO)
    ax.axvline(0, color=TEXTO, lw=0.8)
    ax.set_yticks(y, [p[2] for p in pares], fontsize=8)
    ax.set_xlabel("Beneficio neto esperado frente al diseño base (billones de 2025)")
    ax.set_title("Fase II: líneas futuras subterráneas frente a una variante elevada")
    ax.legend(loc="lower right", fontsize=7)
    guardar(fig, rutas.FIGURAS / "fig_metro_tipologia.pdf")


if __name__ == "__main__":
    abanico_2013()
    beneficios_tasa()
    mapa_decision()
    tipologia_fase2()
