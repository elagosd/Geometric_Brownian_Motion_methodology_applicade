"""Opciones reales: cuándo construir una línea de metro bajo incertidumbre.

La decisión de invertir en un proyecto irreversible cuyos beneficios siguen
un proceso estocástico es una opción americana (McDonald y Siegel, 1986;
Dixit y Pindyck, 1994). Aquí se valora con mínimos cuadrados de Monte Carlo
(Longstaff y Schwartz, 2001) sobre las trayectorias del modelo de red:

1. Para cada escenario se calcula el beneficio anual potencial del proyecto,
   ``B_t = C_t(sin proyecto) - C_t(con proyecto)``, evaluando la red con y
   sin la línea en todos los años.
2. Decidir en el año ``tau`` implica construir en ``[tau, tau + n_c)``,
   abrir en ``tau + n_c`` y recibir ``B_t`` desde entonces, menos O&M,
   más el valor residual en el horizonte.
3. Como el beneficio futuro no se observa al decidir, tanto el valor de
   ejercer como el de continuar se estiman por regresión sobre la variable
   de estado observable (el beneficio potencial corriente).

Se incluye la solución analítica de McDonald-Siegel para validar el
algoritmo en un caso donde existe forma cerrada.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evaluacion import Diseno, evaluar, factores_descuento


@dataclass
class ResultadoOpcion:
    anios_decision: np.ndarray
    vpn_por_anio: np.ndarray  # (K,) media del VPN de decidir en cada año
    valor_flexible: float  # valor con la regla óptima estimada (LSM)
    valor_flexible_ee: float
    valor_ahora: float  # comprometerse en el primer año de decisión
    mejor_anio_estatico: int
    valor_estatico: float
    anio_ejercicio: np.ndarray  # (n,) año de decisión o 0 si nunca
    frontera: np.ndarray  # (K,) beneficio potencial mínimo para ejercer
    prob_nunca: float

    @property
    def valor_opcion(self) -> float:
        """Valor de la flexibilidad frente a la mejor regla sin flexibilidad."""
        return self.valor_flexible - max(self.valor_ahora, self.valor_estatico, 0.0)


def beneficios_potenciales(p, diseno: Diseno, proyecto: str, escenarios, sobrecostos,
                           paso: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Beneficio operativo anual ``(n, Y)`` de tener el proyecto abierto."""
    sin = Diseno(tuple(x for x in diseno.proyectos if x != proyecto), diseno.theta)
    con = Diseno(tuple(sorted(set(sin.proyectos) | {proyecto})), diseno.theta,
                 aperturas=((proyecto, int(escenarios.anios[0])),))
    r_sin = evaluar(p, sin, escenarios, sobrecostos, paso=paso)
    r_con = evaluar(p, con, escenarios, sobrecostos, paso=paso)
    oper = ("tiempo", "auto", "bus")
    b = sum(r_sin.componentes[k] for k in oper) - sum(r_con.componentes[k] for k in oper)
    return r_sin.anios, b


def vpn_decidir(p, proyecto: str, anios: np.ndarray, beneficio: np.ndarray, tau: int,
                multiplicador: np.ndarray) -> np.ndarray:
    """VPN por escenario (a 2026) de decidir construir en ``tau``."""
    e = p.economia
    fila = p.proyectos.set_index("id").loc[proyecto]
    n_c = int(fila["anios_construccion"])
    capex = fila["capex_billones"] * multiplicador
    apertura = tau + n_c
    desc = factores_descuento(p, anios)
    abierto = anios >= apertura
    flujo = (beneficio - e.om_ferro_frac * fila["capex_billones"]) * abierto[None, :]
    vpn = flujo @ desc
    construccion = (anios >= tau) & (anios < apertura)
    vpn -= capex * desc[construccion].sum() / n_c
    vida_usada = max(0, int(anios[-1]) - apertura + 1)
    resto = max(0.0, 1.0 - vida_usada / e.vida_util)
    if apertura <= anios[-1]:
        vpn += capex * resto * desc[-1]
    return vpn


def _base_polinomial(x: np.ndarray, grado: int = 3) -> np.ndarray:
    escala = np.mean(np.abs(x)) + 1e-12
    z = x / escala
    return np.column_stack([z**k for k in range(grado + 1)])


def lsm(p, proyecto: str, anios: np.ndarray, beneficio: np.ndarray, multiplicador: np.ndarray,
        anios_decision: np.ndarray, grado: int = 3) -> ResultadoOpcion:
    """Mínimos cuadrados de Monte Carlo con valor de ejercicio estimado."""
    n = beneficio.shape[0]
    K = anios_decision.size
    vpn = np.stack([vpn_decidir(p, proyecto, anios, beneficio, int(t), multiplicador)
                    for t in anios_decision], axis=1)  # (n, K)
    estado = np.stack([beneficio[:, np.searchsorted(anios, t)] for t in anios_decision], axis=1)
    V = np.zeros(n)
    ejercicio = np.zeros(n, dtype=int)
    frontera = np.full(K, np.nan)
    for k in range(K - 1, -1, -1):
        X = _base_polinomial(estado[:, k], grado)
        coef_e, *_ = np.linalg.lstsq(X, vpn[:, k], rcond=None)
        ejer_hat = X @ coef_e
        if k == K - 1:
            cont_hat = np.zeros(n)
            coef_c = np.zeros_like(coef_e)
        else:
            coef_c, *_ = np.linalg.lstsq(X, V, rcond=None)
            cont_hat = X @ coef_c
        decide = ejer_hat > np.maximum(cont_hat, 0.0)
        V = np.where(decide, vpn[:, k], V)
        ejercicio = np.where(decide, int(anios_decision[k]), ejercicio)
        # Frontera: menor beneficio potencial en el que conviene ejercer.
        malla = np.linspace(estado[:, k].min(), estado[:, k].max(), 400)
        Xm = _base_polinomial(np.concatenate([estado[:, k], malla]), grado)[n:]
        dif = Xm @ coef_e - np.maximum(Xm @ coef_c, 0.0)
        pos = np.nonzero(dif > 0)[0]
        if pos.size:
            frontera[k] = malla[pos[0]]
    medias = vpn.mean(axis=0)
    k_est = int(np.argmax(medias))
    return ResultadoOpcion(
        anios_decision=anios_decision, vpn_por_anio=medias, valor_flexible=float(V.mean()),
        valor_flexible_ee=float(V.std(ddof=1) / np.sqrt(n)), valor_ahora=float(medias[0]),
        mejor_anio_estatico=int(anios_decision[k_est]), valor_estatico=float(medias[k_est]),
        anio_ejercicio=ejercicio, frontera=frontera, prob_nunca=float(np.mean(ejercicio == 0)))


# ----------------------------------------------------------------------
# Validación con la solución analítica de McDonald y Siegel (1986)
# ----------------------------------------------------------------------
def mcdonald_siegel(V0: float, I: float, mu: float, sigma: float, r: float) -> tuple[float, float, float]:
    """Valor de la opción perpetua de invertir ``I`` en un activo ``V`` con MBG.

    Devuelve ``(F(V0), V*, beta)`` con ``beta`` la raíz positiva de
    ``sigma^2/2 b(b-1) + mu b - r = 0`` y el umbral ``V* = beta/(beta-1) I``.
    """
    if mu >= r:
        raise ValueError("Se requiere mu < r para que la opción sea finita.")
    a = 0.5 - mu / sigma**2
    beta = a + np.sqrt(a**2 + 2 * r / sigma**2)
    v_star = beta / (beta - 1.0) * I
    F = (v_star - I) * (V0 / v_star) ** beta if V0 < v_star else V0 - I
    return float(F), float(v_star), float(beta)


def lsm_mcdonald_siegel(V0: float, I: float, mu: float, sigma: float, r: float,
                        horizonte: float = 60.0, dt: float = 0.1, n: int = 20000,
                        grado: int = 3, semilla: int = 3) -> tuple[float, float]:
    """LSM estándar para la opción de invertir con ``V`` MBG (validación)."""
    rng = np.random.default_rng(semilla)
    pasos = int(round(horizonte / dt))
    z = rng.standard_normal((n // 2, pasos))
    z = np.concatenate([z, -z])
    logv = np.log(V0) + np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z, axis=1)
    V = np.exp(logv)
    disc = np.exp(-r * dt)
    flujo = np.maximum(V[:, -1] - I, 0.0)
    for k in range(pasos - 2, -1, -1):
        flujo *= disc
        pago = V[:, k] - I
        dentro = pago > 0
        if dentro.sum() > grado + 1:
            X = _base_polinomial(V[dentro, k] / I, grado)
            coef, *_ = np.linalg.lstsq(X, flujo[dentro], rcond=None)
            cont = X @ coef
            ejerce = pago[dentro] > cont
            idx = np.nonzero(dentro)[0][ejerce]
            flujo[idx] = pago[idx]
    flujo *= disc
    valor = max(float(flujo.mean()), V0 - I)
    return valor, float(flujo.std(ddof=1) / np.sqrt(n))
