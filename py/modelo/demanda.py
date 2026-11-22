"""Generación y distribución de viajes en la hora pico.

Generación:  P_i(t) = V0 * g_i * X_i(t) * (1 - kappa * theta * r(t))
Atracción:   a_j(t) = a_j + rho * theta * r(t) * a_0 * (w_j - 1{j = 0})
Distribución (gravitacional doblemente acotada, Furness):

    T_ij = alpha_i P_i  beta_j A_j  exp(-beta_g c_ij),

con ``c_ij`` el costo compuesto (logsuma) de auto y TP a condiciones de
referencia. ``theta`` es la intensidad de la política de proximidad y
policentrismo y ``r(t)`` su rampa de implementación.
"""

from __future__ import annotations

import numpy as np


def rampa(anio: int, inicio: int, pleno: int) -> float:
    """Fracción de implementación de la política de suelo en ``anio``."""
    if anio <= inicio:
        return 0.0
    if anio >= pleno:
        return 1.0
    return (anio - inicio) / (pleno - inicio)


def generacion(p, indices: np.ndarray, anio: int, theta: float) -> np.ndarray:
    """Producciones motorizadas ``(n, Z)`` en la hora pico."""
    s = p.suelo
    reduccion = 1.0 - s.reduccion_motorizada_max * theta * rampa(anio, s.anio_inicio, s.anio_pleno)
    return p.demanda.viajes_pico_base * p.participacion_generacion[None, :] * indices * reduccion


def atraccion(p, anio: int, theta: float) -> np.ndarray:
    """Participaciones de atracción ``(Z,)`` con la relocalización policéntrica."""
    z = p.zonas
    a = z["atraccion"].to_numpy(float).copy()
    a = a / a.sum()
    w = z["peso_policentrico"].to_numpy(float)
    w = w / w.sum() if w.sum() > 0 else w
    s = p.suelo
    traslado = s.relocalizacion_centro_max * theta * rampa(anio, s.anio_inicio, s.anio_pleno) * a[0]
    a[0] -= traslado
    a += traslado * w
    return a


def costo_compuesto(p, t_ref_via: np.ndarray, t_ref_tp: np.ndarray, dist_via: np.ndarray,
                    n_arcos_tp: np.ndarray) -> np.ndarray:
    """Logsuma de los costos generalizados de referencia (min), forma ``(Z, Z)``."""
    r, e = p.red, p.economia
    Z = p.n_zonas
    cop_min = e.vtts_cop_h / 60.0
    gc_auto = t_ref_via + r.acceso_auto + (e.costo_monetario_auto_km * dist_via + e.costo_fijo_auto) / cop_min
    transb = np.maximum(n_arcos_tp - 1, 0) * 0.5 * r.transbordo
    gc_tp = t_ref_tp + r.espera_acceso_tp + transb + e.tarifa_tp / cop_min - r.asc_tp
    intra = np.eye(Z, dtype=bool).ravel()
    gc_auto = np.where(intra, r.intrazonal_auto + e.costo_fijo_auto / cop_min, gc_auto)
    gc_tp = np.where(intra, r.intrazonal_tp + e.tarifa_tp / cop_min - r.asc_tp, gc_tp)
    lam = r.lambda_logit
    m = np.minimum(gc_auto, gc_tp)
    ls = m - np.log(np.exp(-lam * (gc_auto - m)) + np.exp(-lam * (gc_tp - m))) / lam
    return ls.reshape(Z, Z)


def furness(P: np.ndarray, A: np.ndarray, f: np.ndarray, iteraciones: int = 40,
            tol: float = 1e-9) -> np.ndarray:
    """Modelo gravitacional doblemente acotado, vectorizado en escenarios.

    ``P`` y ``A`` tienen forma ``(n, Z)`` con ``sum(P) = sum(A)`` por fila;
    ``f`` es la función de impedancia ``(Z, Z)``. Devuelve ``T`` de forma
    ``(n, Z, Z)`` con márgenes ``T.sum(2) = P`` y ``T.sum(1) = A``.

    Se escribe ``T_ij = r_i f_ij c_j`` y se alternan
    ``r_i = P_i / sum_j f_ij c_j`` y ``c_j = A_j / sum_i r_i f_ij``.
    """
    c = A.copy()
    for _ in range(iteraciones):
        r = P / np.maximum(c @ f.T, 1e-300)
        c_nuevo = A / np.maximum(r @ f, 1e-300)
        cambio = np.max(np.abs(c_nuevo - c) / np.maximum(np.abs(c), 1e-300))
        c = c_nuevo
        if cambio < tol:
            break
    r = P / np.maximum(c @ f.T, 1e-300)
    return r[:, :, None] * f[None, :, :] * c[:, None, :]


def matriz_od(p, oferta, indices: np.ndarray, anio: int, theta: float) -> np.ndarray:
    """Matriz origen-destino motorizada ``(n, Z, Z)`` para un año."""
    P = generacion(p, indices, anio, theta)
    a = atraccion(p, anio, theta)
    A = P.sum(axis=1, keepdims=True) * a[None, :]
    c = costo_compuesto(p, oferta.t_ref_via, oferta.t_ref_tp, oferta.dist_via, oferta.n_arcos_tp)
    f = np.exp(-p.red.beta_gravedad * (c - c.min()))
    return furness(P, A, f)
