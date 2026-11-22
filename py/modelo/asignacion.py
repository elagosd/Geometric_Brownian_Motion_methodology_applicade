"""Partición modal con congestión: equilibrio por promedios sucesivos (MSA).

Dada la matriz OD motorizada ``T`` (personas en la hora pico), se busca la
participación del TP ``s_ij`` que es punto fijo de

    s = Logit(GC_tp(s) - GC_auto(s)),

donde los tiempos dependen de las cargas por corredor:

    vía:  t_e = t0_e (1 + alpha (V_e / C_e)^beta)              (BPR)
    TP:   t_e = L_e h_e (1 + a (Q_e / K_e)^b)                    (hacinamiento)

con ``h_e`` la media armónica de las velocidades de los servicios del
corredor (el bus zonal a una velocidad exógena de referencia). Como el costo
de cada modo depende solo de su propia carga y las funciones son crecientes,
el problema equivale a un programa convexo y el equilibrio es único.
Se resuelve con promedios sucesivos de paso autorregulado.
Todo se vectoriza en los escenarios: ``T`` tiene forma ``(n, Z, Z)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ResultadoAsignacion:
    s_tp: np.ndarray  # (n, Z*Z)
    tiempo_auto: np.ndarray  # (n, Z*Z) min puerta a puerta
    tiempo_tp: np.ndarray  # (n, Z*Z) min percibidos (incluye hacinamiento)
    carga_tp: np.ndarray  # (n, D) pasajeros/h
    flujo_via: np.ndarray  # (n, D) veh/h
    vel_via: np.ndarray  # (n, D) km/h
    hacinamiento: np.ndarray  # (n, D) Q/K
    horas_persona: np.ndarray  # (n,) en la hora pico
    horas_tp: np.ndarray
    horas_auto: np.ndarray
    veh_km: np.ndarray  # (n,)
    pax_km_bus: np.ndarray  # (n,)
    viajes: np.ndarray  # (n,)
    viajes_tp: np.ndarray  # (n,)
    brecha: float  # cambio relativo de s en la última iteración

    @property
    def tiempo_medio(self) -> np.ndarray:
        return self.horas_persona * 60.0 / self.viajes

    @property
    def participacion_tp(self) -> np.ndarray:
        return self.viajes_tp / self.viajes

    def velocidad_media_via(self, longitud: np.ndarray) -> np.ndarray:
        """Velocidad media ponderada por veh-km (km/h)."""
        vkm = self.flujo_via * longitud[None, :]
        horas = vkm / self.vel_via
        return vkm.sum(1) / np.maximum(horas.sum(1), 1e-12)


def funcion_congestion(x: np.ndarray, alfa: float, beta: float, x_lim: float,
                       f_max: float) -> np.ndarray:
    """Multiplicador del tiempo libre: potencia hasta ``x_lim``, tangente después.

    ``f(x) = 1 + alfa x^beta`` si ``x <= x_lim``; si no,
    ``f(x_lim) + alfa beta x_lim^(beta-1) (x - x_lim)``; y siempre ``f <= f_max``.
    La función es continua, creciente y convexa hasta la cota.
    """
    f_lim = 1.0 + alfa * x_lim**beta
    pendiente = alfa * beta * x_lim ** (beta - 1.0)
    f = np.where(x <= x_lim, 1.0 + alfa * np.minimum(x, x_lim) ** beta,
                 f_lim + pendiente * (x - x_lim))
    return np.minimum(f, f_max)


def costos_fijos_od(p, oferta):
    """Componentes del costo generalizado que no dependen de la congestión."""
    r, e = p.red, p.economia
    Z = oferta.Z
    cop_min = e.vtts_cop_h / 60.0
    intra = np.eye(Z, dtype=bool).ravel()
    dinero_auto = (e.costo_monetario_auto_km * oferta.dist_via + e.costo_fijo_auto) / cop_min
    dinero_tp = np.full(Z * Z, e.tarifa_tp / cop_min)
    transb = np.maximum(oferta.n_arcos_tp - 1, 0) * 0.5 * r.transbordo
    extra_auto = np.where(intra, r.intrazonal_auto, r.acceso_auto)
    extra_tp = np.where(intra, r.intrazonal_tp, r.espera_acceso_tp + transb)
    return intra, dinero_auto, dinero_tp, extra_auto, extra_tp


def equilibrio(p, oferta, T: np.ndarray, s0: np.ndarray | None = None,
               iteraciones: int | None = None) -> ResultadoAsignacion:
    r, d = p.red, p.demanda
    n = T.shape[0]
    Z = oferta.Z
    Tf = T.reshape(n, Z * Z)
    intra, dinero_auto, dinero_tp, extra_auto, extra_tp = costos_fijos_od(p, oferta)
    L = oferta.longitud
    t0 = L / r.vel_libre_via * 60.0
    # Horas por km en el TP sin hacinamiento: media armónica de los servicios,
    # con el bus zonal a velocidad exógena. Así el costo de cada modo depende
    # solo de su propia carga y el equilibrio es único.
    h_km = (oferta.tp_fijo + oferta.cap_zonal / r.vel_ref_zonal) / oferta.cap_tp
    lam = r.lambda_logit
    k_max = iteraciones or r.iter_msa

    def tiempos(s):
        Tt = Tf * s
        Ta = Tf * (1.0 - s)
        carga = Tt @ oferta.A_tp.T  # (n, D)
        flujo = (Ta @ oferta.A_via.T) / d.ocupacion_auto
        x_via = flujo / oferta.cap_via
        factor = funcion_congestion(x_via, r.bpr_alfa, r.bpr_beta, r.bpr_x_lim,
                                    r.vel_libre_via / r.vel_min_via)
        t_via = t0 * factor
        vel = r.vel_libre_via / factor
        x_tp = carga / oferta.cap_tp
        t_tp = L * h_km * 60.0 * funcion_congestion(x_tp, r.hac_alfa, r.hac_beta,
                                                    r.hac_x_lim, r.hac_f_max)
        tau_a = np.where(intra, 0.0, t_via @ oferta.A_via) + extra_auto
        tau_t = np.where(intra, 0.0, t_tp @ oferta.A_tp) + extra_tp
        return tau_a, tau_t, carga, flujo, vel, x_tp, Tt, Ta

    if s0 is None:
        gc_a = oferta.t_ref_via + extra_auto + dinero_auto
        gc_t = oferta.t_ref_tp + extra_tp + dinero_tp - r.asc_tp
        s = np.broadcast_to(1.0 / (1.0 + np.exp(lam * (gc_t - gc_a))), (n, Z * Z)).copy()
    else:
        s = s0.copy()

    # Promedios con paso autorregulado: se parte de un paso grande y se reduce
    # a la mitad cada vez que la brecha crece (Liu, He y He, 2009).
    brecha = np.inf
    paso_k = r.paso_inicial
    for k in range(1, k_max + 1):
        tau_a, tau_t, *_ = tiempos(s)
        gc_a = tau_a + dinero_auto
        gc_t = tau_t + dinero_tp - r.asc_tp
        s_obj = 1.0 / (1.0 + np.exp(np.clip(lam * (gc_t - gc_a), -50, 50)))
        nueva = float(np.sum(np.abs(s_obj - s) * Tf) / np.sum(Tf))
        if nueva > brecha:
            paso_k = max(0.5 * paso_k, r.paso_minimo)
        brecha = nueva
        if brecha < r.tol_equilibrio:
            break
        s = s + paso_k * (s_obj - s)

    tau_a, tau_t, carga, flujo, vel, x_tp, Tt, Ta = tiempos(s)
    horas_tp = (Tt * tau_t).sum(1) / 60.0
    horas_auto = (Ta * tau_a).sum(1) / 60.0
    veh_km = (flujo * L).sum(1)
    pax_km_bus = (carga * L * oferta.frac_bus).sum(1)
    res = ResultadoAsignacion(
        s_tp=s, tiempo_auto=tau_a, tiempo_tp=tau_t, carga_tp=carga, flujo_via=flujo,
        vel_via=vel, hacinamiento=x_tp, horas_persona=horas_tp + horas_auto,
        horas_tp=horas_tp, horas_auto=horas_auto, veh_km=veh_km, pax_km_bus=pax_km_bus,
        viajes=Tf.sum(1), viajes_tp=Tt.sum(1), brecha=brecha)
    return res
