"""Cálculos de la aplicación con memoria (envoltorios del paquete ``modelo``)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "py"))

from modelo import gbm  # noqa: E402
from modelo.evaluacion import Diseno, evaluar, muestrear_sobrecostos  # noqa: E402
from modelo.opciones_reales import (beneficios_potenciales, lsm, lsm_mcdonald_siegel,  # noqa: E402
                                    mcdonald_siegel)
from modelo.optimizacion import enumerar_disenos, resumen  # noqa: E402
from modelo.parametros import FUENTES, cargar_serie_transmilenio, parametros_calibrados  # noqa: E402

BASE = "Base | θ=0"

CONFIG_INICIAL = {
    "n": 256, "metodo": "sobol", "semilla": 2026, "paso": 2,
    "sigma_comun": 0.05, "mu_comun": 0.005, "vtts": 8000.0, "tasa": 0.09,
    "sobrecostos": True, "saltos": False, "lam_saltos": 0.04, "media_salto": -0.35,
    "costo_suelo": 0.6, "reduccion_nm": 0.05, "relocalizacion": 0.15,
}


@st.cache_resource(show_spinner=False)
def parametros_base():
    return parametros_calibrados()


def clave(cfg: dict) -> tuple:
    return tuple(sorted(cfg.items()))


def parametros(cfg_t: tuple):
    cfg = dict(cfg_t)
    p = parametros_base()
    p = p.ajustar("demanda", sigma_comun=cfg["sigma_comun"], mu_comun=cfg["mu_comun"],
                  intensidad_saltos=cfg["lam_saltos"] if cfg["saltos"] else 0.0,
                  media_salto=cfg["media_salto"])
    p = p.ajustar("economia", vtts_cop_h=cfg["vtts"], tasa_descuento=cfg["tasa"])
    p = p.ajustar("sobrecostos", activar=cfg["sobrecostos"])
    p = p.ajustar("suelo", costo_anual_billones=cfg["costo_suelo"],
                  reduccion_motorizada_max=cfg["reduccion_nm"], relocalizacion_centro_max=cfg["relocalizacion"])
    return p


@st.cache_data(show_spinner=False, max_entries=8)
def escenarios(cfg_t: tuple):
    cfg = dict(cfg_t)
    p = parametros(cfg_t)
    return gbm.escenarios_desde_parametros(p, cfg["n"], cfg["metodo"], cfg["semilla"])


@st.cache_data(show_spinner=False, max_entries=8)
def sobrecostos(cfg_t: tuple):
    return muestrear_sobrecostos(parametros(cfg_t), dict(cfg_t)["n"])


@st.cache_data(show_spinner=False, max_entries=64)
def evaluar_diseno(cfg_t: tuple, proyectos: tuple, theta: float):
    p = parametros(cfg_t)
    return evaluar(p, Diseno(tuple(sorted(proyectos)), float(theta)), escenarios(cfg_t),
                   sobrecostos(cfg_t), paso=dict(cfg_t)["paso"])


def evaluar_todos(cfg_t: tuple, barra=None):
    p = parametros(cfg_t)
    disenos = enumerar_disenos(p)
    res = {}
    for k, d in enumerate(disenos):
        if barra is not None:
            barra.progress((k + 1) / len(disenos), text=f"Evaluando {d.nombre} ({k + 1}/{len(disenos)})")
        res[d.nombre] = evaluar_diseno(cfg_t, d.proyectos, d.theta)
    return res


@st.cache_data(show_spinner=False, max_entries=8)
def tabla_resumen(cfg_t: tuple):
    res = evaluar_todos(cfg_t)
    df = resumen(res, BASE)
    vp = {k: v.vp for k, v in res.items()}
    return df, vp


@st.cache_data(show_spinner=False, max_entries=8)
def opcion_real(cfg_t: tuple, proyecto: str, otros: tuple, theta: float, primer: int, ultimo: int):
    p = parametros(cfg_t)
    esc, sc = escenarios(cfg_t), sobrecostos(cfg_t)
    d = Diseno(tuple(sorted(set(otros) | {proyecto})), theta)
    anios, B = beneficios_potenciales(p, d, proyecto, esc, sc, paso=dict(cfg_t)["paso"])
    res = lsm(p, proyecto, anios, B, sc[proyecto], np.arange(primer, ultimo + 1))
    return anios, B, res


@st.cache_data(show_spinner=False)
def validacion_ms(V0, I, mu, sigma, r, n):
    return mcdonald_siegel(V0, I, mu, sigma, r), lsm_mcdonald_siegel(V0, I, mu, sigma, r, horizonte=80, dt=0.1, n=n)


@st.cache_data(show_spinner=False)
def convergencia(cfg_t: tuple, tamanos: tuple, reps: int):
    p = parametros(cfg_t)
    filas = []
    for metodo in ("mc", "antitetica", "sobol"):
        for n in tamanos:
            est = []
            for r in range(reps):
                esc = gbm.escenarios_desde_parametros(p, n, metodo, semilla=500 + r)
                est.append(evaluar(p, Diseno(), esc, muestrear_sobrecostos(p, n), paso=5).vp.mean())
            filas.append({"metodo": metodo, "n": n, "media": np.mean(est), "desv": np.std(est, ddof=1)})
    return pd.DataFrame(filas)


__all__ = ["FUENTES", "cargar_serie_transmilenio", "gbm", "Diseno"]


# ---------------------------------------------------------------------------
# Fase I (2013) y metro elevado vs. subterráneo: resultados de py/correr_fase_2013.py
# ---------------------------------------------------------------------------
def _carpeta_resultados() -> Path:
    for c in (RAIZ / "salidas", RAIZ / "resultados"):
        if (c / "fase2013_exante.csv").exists():
            return c
    raise FileNotFoundError("Ejecute py/correr_fase_2013.py para generar los resultados de la fase I.")


@st.cache_data(show_spinner=False)
def resultados_fase1() -> dict:
    import json
    c = _carpeta_resultados()
    out = {k: pd.read_csv(c / f"{k}.csv") for k in
           ("fase2013_exante", "fase2013_expost", "fase2013_descuento", "fase2013_umbral", "fase2_tipologia", "metro_costos")}
    out["resumen"] = json.loads((c / "fase2013_resumen.json").read_text(encoding="utf-8"))
    out["calibracion"] = json.loads((c / "fase2013_calibracion.json").read_text(encoding="utf-8"))
    npz = c / "fase2013_escenarios.npz"
    if npz.exists():
        z = np.load(npz)
        out["escenarios"] = {"anios": z["anios"], "total": z["total"], "observado": z["observado"]}
    return out


def costos_metro(km: float, estaciones: int) -> dict:
    from modelo import metro
    return metro.comparar(km, estaciones)
