"""Calibración del año base (2025).

La calibración tiene dos niveles.

1. Capacidades efectivas por corredor. Los datos de capacidad por corredor
   son aproximados, y una asignación de camino único concentra los flujos.
   Por eso, en el año base, la capacidad vial de cada corredor se hace
   proporcional a su flujo en el sentido de mayor carga (con un piso), y la
   capacidad del TP se eleva, si hace falta, hasta la carga observada en el
   modelo (el sistema transporta hoy esa demanda, al límite de su
   capacidad). Así el año base parte de una relación volumen/capacidad
   homogénea y realista.

2. Parámetros globales. Con esas capacidades, se ajustan por mínimos
   cuadrados no lineales tres parámetros para tres metas observadas:

       escala de capacidad vial   ->  velocidad en hora pico de 18,9 km/h
       constante modal del TP     ->  participación del TP de 35/60
       parámetro gravitacional    ->  duración media de 48 minutos

Ambos niveles se alternan hasta que las capacidades dejan de cambiar.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
from scipy.optimize import least_squares

from .asignacion import equilibrio
from .demanda import matriz_od
from .parametros import DIR_DATOS, Parametros
from .red import abiertos_en, construir_oferta, limpiar_cache

ARCHIVO = DIR_DATOS / "calibracion.json"
PISO_PESO_VIA = 0.25
HACINAMIENTO_BASE = 1.0


def _resolver_base(p: Parametros, iteraciones: int = 300):
    anio = p.demanda.anio_base
    oferta = construir_oferta(p, abiertos_en(p, anio, ()))
    T = matriz_od(p, oferta, np.ones((1, p.n_zonas)), anio, theta=0.0)
    q = p.ajustar("red", tol_equilibrio=1e-7)
    res = equilibrio(q, oferta, T, iteraciones=iteraciones)
    return oferta, T, res


def indicadores_base(p: Parametros, iteraciones: int = 300) -> dict:
    oferta, _, res = _resolver_base(p, iteraciones)
    return {
        "vel_via": float(res.velocidad_media_via(oferta.longitud)[0]),
        "part_tp": float(res.participacion_tp[0]),
        "tiempo_medio": float(res.tiempo_medio[0]),
        "hacinamiento_max": float(res.hacinamiento.max()),
        "vc_via_max": float((res.flujo_via / oferta.cap_via).max()),
        "brecha_msa": res.brecha,
    }


def _ajuste_global(p: Parametros) -> tuple[Parametros, float]:
    m = p.metas

    def construir(x):
        return p.ajustar("red", escala_cap_via=float(np.exp(x[0])), asc_tp=float(x[1]),
                         beta_gravedad=float(np.exp(x[2])))

    def residuos(x):
        q = construir(x)
        limpiar_cache()
        ind = indicadores_base(q)
        return np.array([
            (ind["vel_via"] - m.vel_via) / m.vel_via,
            (ind["part_tp"] - m.part_tp) / m.part_tp,
            (ind["tiempo_medio"] - m.tiempo_medio) / m.tiempo_medio,
        ])

    r = p.red
    x0 = np.array([np.log(r.escala_cap_via), r.asc_tp, np.log(r.beta_gravedad)])
    sol = least_squares(residuos, x0, x_scale=[0.5, 5.0, 0.5], xtol=1e-10, ftol=1e-10)
    return construir(sol.x), float(np.max(np.abs(sol.fun)))


def calibrar(p: Parametros, rondas: int = 10, amortiguamiento: float = 0.5,
             verbose: bool = False) -> tuple[Parametros, dict]:
    """Alterna el ajuste global y la actualización amortiguada de capacidades.

    En cada ronda: (i) se ajustan escala, constante modal y parámetro
    gravitacional; (ii) los pesos viales se mueven hacia el flujo del sentido
    de mayor carga (con piso) y la capacidad del TP hacia
    ``max(dato, carga / HACINAMIENTO_BASE)``. La escala vial de arranque de
    la ronda siguiente se fija para que un corredor con flujo medio tenga la
    relación v/c que, con BPR, produce la velocidad objetivo.
    """
    r, m = p.red, p.metas
    x_obj = ((r.vel_libre_via / m.vel_via - 1.0) / r.bpr_alfa) ** (1.0 / r.bpr_beta)
    q = p
    E = len(p.corredores)
    cap_dato = p.corredores["cap_tp_base"].to_numpy(float)
    historial = []
    for ronda in range(rondas):
        q, residuo = _ajuste_global(q)
        limpiar_cache()
        oferta, _, res = _resolver_base(q)
        flujo = res.flujo_via[0]
        carga = res.carga_tp[0]
        pico_via = np.maximum(flujo[:E], flujo[E:])
        pico_tp = np.maximum(carga[:E], carga[E:])
        objetivo = np.maximum(pico_via / pico_via.mean(), PISO_PESO_VIA)
        objetivo = objetivo / objetivo.mean()
        c = q.corredores.copy()
        viejo = c["cap_via_rel"].to_numpy(float)
        peso = (1 - amortiguamiento) * viejo + amortiguamiento * objetivo
        peso = peso / peso.mean()
        cap_obj = np.maximum(cap_dato, pico_tp / HACINAMIENTO_BASE)
        cap_vieja = c["cap_tp_base"].to_numpy(float)
        cap_nueva = (1 - amortiguamiento) * cap_vieja + amortiguamiento * cap_obj
        historial.append({
            "ronda": ronda, "residuo": residuo,
            "cambio_via": float(np.max(np.abs(peso - viejo) / viejo)),
            "cambio_tp": float(np.max(np.abs(cap_nueva - cap_vieja) / cap_vieja)),
            "hacinamiento_max": float(res.hacinamiento.max()),
        })
        if verbose:
            print(historial[-1])
        c["cap_via_rel"] = peso
        c["cap_tp_base"] = np.round(cap_nueva, 0)
        escala0 = float(pico_via.mean() / x_obj)
        q = q.con(corredores=c).ajustar("red", escala_cap_via=escala0)
        limpiar_cache()
    q, residuo = _ajuste_global(q)
    limpiar_cache()
    ind = indicadores_base(q)
    info = {
        "escala_cap_via": q.red.escala_cap_via,
        "asc_tp": q.red.asc_tp,
        "beta_gravedad": q.red.beta_gravedad,
        "cap_via_rel": q.corredores["cap_via_rel"].round(6).tolist(),
        "cap_tp_base": q.corredores["cap_tp_base"].tolist(),
        "indicadores": ind,
        "metas": asdict(p.metas),
        "residuo_max": residuo,
        "rondas": historial,
    }
    return q, info


def aplicar(p: Parametros, info: dict) -> Parametros:
    c = p.corredores.copy()
    if len(info.get("cap_via_rel", [])) == len(c):
        c["cap_via_rel"] = info["cap_via_rel"]
        c["cap_tp_base"] = info["cap_tp_base"]
    q = p.con(corredores=c)
    return q.ajustar("red", escala_cap_via=info["escala_cap_via"], asc_tp=info["asc_tp"],
                     beta_gravedad=info["beta_gravedad"])


def guardar(info: dict) -> None:
    ARCHIVO.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def cargar_o_calibrar(p: Parametros, forzar: bool = False) -> Parametros:
    if ARCHIVO.exists() and not forzar:
        return aplicar(p, json.loads(ARCHIVO.read_text(encoding="utf-8")))
    q, info = calibrar(p)
    guardar(info)
    return q
