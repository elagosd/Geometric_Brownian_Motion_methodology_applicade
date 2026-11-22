"""Fase I: el modelo con las condiciones de 2013 y la decisión del metro.

En 2013 Bogotá contrató el diseño de una primera línea de metro subterránea
(Portal de las Américas - calle 127, entregado en 2014); en 2016 la decisión
cambió a un metro elevado hasta la calle 72, que es el que se construye. Esta
fase reproduce la decisión desde la información disponible en 2013-2014
(evaluación ex ante) y la contrasta con lo que ocurrió después (evaluación
ex post: el censo de 2018, la pandemia y la apertura en 2028).

Datos de 2013 usados en la calibración:
    población de Bogotá        7.674.366 (DANE, proyección del censo 2005)
    viajes diarios             17,6 millones (Encuesta de Movilidad 2011)
    reparto                    TP 30 %, auto 10 %, taxi 4 % (EM 2011); moto 3 % (supuesto)
    velocidad                  24 km/h promedio diurno (SDM) frente a 23 km/h en 2023
    tasa social de descuento   12 % (vigente hasta la Resolución 1092 de 2022)
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import metro as M
from .asignacion import equilibrio
from .demanda import matriz_od
from .gbm import EscenariosMBG
from .parametros import DIR_DATOS, Metas, Parametros
from .red import abiertos_en, construir_oferta, limpiar_cache

ARCHIVO_CAL_2013 = DIR_DATOS / "calibracion_2013.json"
POB_BOGOTA_2013 = 7674.366  # miles, DANE (proyección con base en el censo 2005)
CRECIMIENTO_REGION = 0.025  # supuesto para retroproyectar los municipios
VIAJES_PICO_2013 = 930_000.0  # 17,6 M * 47 % motorizados * 1,12 región * 10 % pico
VEL_PICO_2013 = 18.9 * 24.0 / 23.0  # velocidad pico 2025 escalada por la razón de promedios diurnos
PART_TP_2013 = 30.0 / 47.0
CRECIMIENTO_BOGOTA_2013 = 0.0127  # DANE 2005: 7,67 M (2013) a 8,38 M (2020)

# Trayectoria observada estilizada del índice agregado de demanda motorizada
# (2013 = 1): crecimiento moderado hasta 2019, choque de 2020 y recuperación
# incompleta (la EM 2023 registra menos viajes que la de 2019).
TRAYECTORIA_OBSERVADA = {
    2013: 1.000, 2014: 1.008, 2015: 1.016, 2016: 1.024, 2017: 1.032, 2018: 1.040, 2019: 1.048,
    2020: 0.700, 2021: 0.860, 2022: 0.930, 2023: 0.940, 2024: 0.945, 2025: 0.946,
}


def proyectos_2013() -> pd.DataFrame:
    """Alternativas de la primera línea de metro vistas desde 2013-2014."""
    l1e = M.capex_total_2025("elevado_factibilidad_2017")
    sub_oficial = M.capex_total_2025("subterraneo_2015")
    c27 = M.comparar(27.0, 28)
    sub_unitarios = c27["capex_elevado"] + c27["ahorro_elevado"]
    filas = [
        dict(id="L1E", nombre="Metro elevado Bosa – calle 72", estado="decision", apertura=2021,
             capex_billones=round(l1e, 2), clase_sobrecosto="ferro", capacidad=72000, velocidad=43,
             aristas="e08:1;e07:1", anios_construccion=5,
             fuente="Factibilidad 2017 (CONPES 3900): 23,96 km, 16 estaciones, 72.000 pas/h-sentido, 43 km/h; CAPEX EMB indexado"),
        dict(id="L1S", nombre="Metro subterráneo Américas – calle 127 (costo oficial)", estado="decision",
             apertura=2023, capex_billones=round(sub_oficial, 2), clase_sobrecosto="ferro", capacidad=80000,
             velocidad=36, aristas="e08:1;e07:1;e01:0.55", anios_construccion=7,
             fuente="Diseño 2014-2015 (Consorcio L1): ~27 km, 28 estaciones, 80.000 pas/h-sentido; CAPEX EMB indexado; velocidad supuesta"),
        dict(id="L1S_alto", nombre="Metro subterráneo Américas – calle 127 (costo con unitarios 2022)",
             estado="decision", apertura=2023, capex_billones=round(sub_unitarios, 2), clase_sobrecosto="ferro",
             capacidad=80000, velocidad=36, aristas="e08:1;e07:1;e01:0.55", anios_construccion=7,
             fuente="Mismo diseño con los costos unitarios elevado/subterráneo del estudio EMB 2023"),
    ]
    return pd.DataFrame(filas)


def zonas_2013(p: Parametros) -> pd.DataFrame:
    z = p.zonas.copy()
    bog = (z["ambito"] == "Bogotá").to_numpy()
    pob = z["poblacion_miles"].to_numpy(float).copy()
    pob[bog] = pob[bog] / pob[bog].sum() * POB_BOGOTA_2013
    pob[~bog] = pob[~bog] * (1 + CRECIMIENTO_REGION) ** -12
    z["poblacion_miles"] = np.round(pob, 0)
    # Derivas vistas desde 2013: las proyecciones del censo 2005 esperaban
    # que Bogotá creciera 1,27 % anual; se conservan las diferencias entre zonas.
    mu = z["mu_demografica"].to_numpy(float).copy()
    mu_bog_2025 = np.average(mu[bog], weights=pob[bog])
    mu[bog] = mu[bog] - mu_bog_2025 + CRECIMIENTO_BOGOTA_2013
    mu[~bog] = 0.03
    z["mu_demografica"] = np.round(mu, 4)
    return z


def parametros_2013(p25: Parametros, tasa: float = 0.12) -> Parametros:
    """Parámetros de la fase I a partir de la red calibrada en 2025."""
    q = replace(p25, zonas=zonas_2013(p25), proyectos=proyectos_2013(),
                metas=Metas(vel_via=VEL_PICO_2013, part_tp=PART_TP_2013, tiempo_medio=float("nan")))
    q = q.ajustar("demanda", viajes_pico_base=VIAJES_PICO_2013, anio_base=2013, anio_fin=2050, mu_comun=0.01)
    q = q.ajustar("economia", tasa_descuento=tasa, anio_vp=2014)
    q = q.ajustar("suelo", anio_inicio=2015, anio_pleno=2028)
    return q


def _indicadores_2013(q: Parametros) -> dict:
    oferta = construir_oferta(q, abiertos_en(q, 2013, ()))
    T = matriz_od(q, oferta, np.ones((1, q.n_zonas)), 2013, 0.0)
    r = equilibrio(q.ajustar("red", tol_equilibrio=1e-7), oferta, T, iteraciones=300)
    return {"vel_via": float(r.velocidad_media_via(oferta.longitud)[0]),
            "part_tp": float(r.participacion_tp[0]), "tiempo_medio": float(r.tiempo_medio[0]),
            "hacinamiento_max": float(r.hacinamiento.max())}


def calibrar_2013(q: Parametros) -> tuple[Parametros, dict]:
    """Ajusta escala vial y constante modal a 2013; el parámetro gravitacional se conserva."""
    m = q.metas

    def construir(x):
        return q.ajustar("red", escala_cap_via=float(np.exp(x[0])), asc_tp=float(x[1]))

    def residuos(x):
        limpiar_cache()
        ind = _indicadores_2013(construir(x))
        return np.array([(ind["vel_via"] - m.vel_via) / m.vel_via, (ind["part_tp"] - m.part_tp) / m.part_tp])

    sol = least_squares(residuos, [np.log(q.red.escala_cap_via), q.red.asc_tp], x_scale=[0.5, 5.0],
                        xtol=1e-10, ftol=1e-10)
    r = construir(sol.x)
    limpiar_cache()
    ind = _indicadores_2013(r)
    info = {"escala_cap_via": r.red.escala_cap_via, "asc_tp": r.red.asc_tp, "indicadores": ind,
            "metas": {"vel_via": m.vel_via, "part_tp": m.part_tp}, "residuo_max": float(np.max(np.abs(sol.fun)))}
    return r, info


def cargar_o_calibrar_2013(p25: Parametros, forzar: bool = False, tasa: float = 0.12) -> Parametros:
    q = parametros_2013(p25, tasa)
    if ARCHIVO_CAL_2013.exists() and not forzar:
        info = json.loads(ARCHIVO_CAL_2013.read_text(encoding="utf-8"))
        return q.ajustar("red", escala_cap_via=info["escala_cap_via"], asc_tp=info["asc_tp"])
    r, info = calibrar_2013(q)
    ARCHIVO_CAL_2013.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return r


def escenarios_expost(p13: Parametros, esc25: EscenariosMBG) -> EscenariosMBG:
    """Trayectoria observada 2013-2025 seguida de los escenarios de la fase II.

    El índice agregado sigue ``TRAYECTORIA_OBSERVADA``; las zonas de la región
    crecen 1,5 % anual más que las de Bogotá, y el conjunto se reescala para
    reproducir el agregado observado. Desde 2026 cada escenario multiplica el
    nivel de 2025 por el índice simulado en la fase II.
    """
    anios_obs = np.arange(2013, 2026)
    agregado = np.array([TRAYECTORIA_OBSERVADA[a] for a in anios_obs])
    region = (p13.zonas["ambito"] != "Bogotá").to_numpy()
    pesos = p13.participacion_generacion
    rel = np.exp(np.outer(anios_obs - 2013, np.where(region, 0.015, 0.0)))
    X_obs = rel * (agregado / (rel @ pesos))[:, None]
    n = esc25.n
    fut = esc25.X[:, 1:, :] * X_obs[-1][None, None, :]
    X = np.concatenate([np.broadcast_to(X_obs, (n,) + X_obs.shape), fut], axis=1)
    anios = np.concatenate([anios_obs, esc25.anios[1:]])
    return EscenariosMBG(X, anios, esc25.mu, esc25.cov, "expost")
