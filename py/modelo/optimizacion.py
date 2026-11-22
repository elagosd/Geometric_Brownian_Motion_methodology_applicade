"""Enumeración de diseños urbanos y criterios de decisión bajo riesgo.

El problema es un programa estocástico de dos etapas aproximado por
promedio muestral (SAA): con ``n`` escenarios comunes a todos los diseños,

    min_{d in D}  (1 - w) E_n[VP_d] + w CVaR_{a,n}[VP_d]
    s. a.         E_n[CAPEX_d] <= B,

donde ``D`` contiene todas las combinaciones de proyectos en decisión y
los niveles de la política de suelo. Como ``D`` es pequeño (48 diseños), se
resuelve por enumeración exacta; la incertidumbre de muestreo se reporta
con errores estándar e intervalos bootstrap.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd

from .evaluacion import Diseno, ResultadoDiseno, evaluar
from .riesgo import cvar, error_estandar, frontera_pareto, var

NIVELES_THETA = (0.0, 0.5, 1.0)


def enumerar_disenos(p, thetas=NIVELES_THETA) -> list[Diseno]:
    dec = p.proyectos_decision
    disenos = []
    for k in range(len(dec) + 1):
        for combo in combinations(dec, k):
            for th in thetas:
                disenos.append(Diseno(tuple(combo), float(th)))
    return disenos


def evaluar_todos(p, disenos: list[Diseno], escenarios, sobrecostos, paso: int = 1,
                  progreso: Callable[[int, int, str], None] | None = None) -> dict[str, ResultadoDiseno]:
    resultados = {}
    for k, d in enumerate(disenos):
        if progreso:
            progreso(k, len(disenos), d.nombre)
        resultados[d.nombre] = evaluar(p, d, escenarios, sobrecostos, paso=paso)
    if progreso:
        progreso(len(disenos), len(disenos), "listo")
    return resultados


def resumen(resultados: dict[str, ResultadoDiseno], base: str = "Base | θ=0",
            alpha: float = 0.95) -> pd.DataFrame:
    vp_base = resultados[base].vp
    filas = []
    for nombre, r in resultados.items():
        vp = r.vp
        bn = vp_base - vp  # beneficio neto frente al diseño base
        filas.append({
            "diseno": nombre,
            "proyectos": "+".join(r.diseno.proyectos) or "—",
            "theta": r.diseno.theta,
            "vp_media": vp.mean(),
            "vp_ee": error_estandar(vp),
            "vp_desv": vp.std(ddof=1),
            "vp_var": var(vp, alpha),
            "vp_cvar": cvar(vp, alpha),
            "bn_media": bn.mean(),
            "bn_ee": error_estandar(bn),
            "prob_bn_pos": float(np.mean(bn > 0)),
            "bn_cvar_perdida": cvar(-bn, alpha),
            "capex_medio": float(r.capex_total.mean()),
            "tiempo_2050": float(r.tiempo_medio[:, -1].mean()),
            "vel_2050": float(r.vel_via[:, -1].mean()),
            "part_tp_2050": float(r.part_tp[:, -1].mean()),
            "co2_2050": float(r.co2_mt[:, -1].mean()),
        })
    df = pd.DataFrame(filas)
    idx = frontera_pareto(df["vp_media"].to_numpy(), df["vp_cvar"].to_numpy())
    df["pareto"] = False
    df.loc[df.index[idx], "pareto"] = True
    return df.sort_values("vp_media").reset_index(drop=True)


def elegir(df: pd.DataFrame, w: float = 0.0, presupuesto: float | None = None) -> pd.Series:
    """Diseño que minimiza ``(1-w) E[VP] + w CVaR`` con restricción de CAPEX."""
    cand = df if presupuesto is None else df[df["capex_medio"] <= presupuesto + 1e-9]
    if cand.empty:
        raise ValueError("Ningún diseño cumple el presupuesto.")
    obj = (1 - w) * cand["vp_media"] + w * cand["vp_cvar"]
    return cand.loc[obj.idxmin()]


def ruta_optima(df: pd.DataFrame, pesos=np.linspace(0, 1, 11)) -> pd.DataFrame:
    """Diseño elegido para cada nivel de aversión al riesgo ``w``."""
    filas = []
    for w in pesos:
        e = elegir(df, w)
        filas.append({"w": w, "diseno": e["diseno"], "vp_media": e["vp_media"], "vp_cvar": e["vp_cvar"]})
    return pd.DataFrame(filas)
