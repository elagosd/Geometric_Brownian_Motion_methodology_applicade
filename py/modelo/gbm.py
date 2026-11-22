"""Movimiento browniano geométrico (MBG) multivariado.

Cada zona ``i`` tiene un índice de demanda ``X_i(t)`` con ``X_i(0) = 1`` que
sigue

    dX_i / X_i = mu_i dt + sum_k L_ik dW_k + (e^J - 1) dN,

donde ``L`` es el factor de Cholesky de la matriz de covarianza instantánea
``Sigma`` y ``N`` es un proceso de Poisson de saltos sistémicos (opcional).
La discretización es exacta en distribución:

    ln X(t+dt) = ln X(t) + (mu - diag(Sigma)/2) dt + L Z sqrt(dt) + J,

por lo que no hay sesgo de discretización: el error es solo de muestreo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.stats import qmc

METODOS = ("mc", "antitetica", "sobol")


def covarianza_factorial(sigma_comun: float, sigma_idio: np.ndarray,
                         cargas: np.ndarray | None = None) -> np.ndarray:
    """Covarianza de un modelo de un factor: ``Sigma = s_c^2 b b' + diag(s^2)``."""
    sigma_idio = np.asarray(sigma_idio, float)
    b = np.ones_like(sigma_idio) if cargas is None else np.asarray(cargas, float)
    return sigma_comun**2 * np.outer(b, b) + np.diag(sigma_idio**2)


def correlacion(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def normales(n: int, dim: int, metodo: str = "mc", semilla: int | None = None) -> np.ndarray:
    """Matriz ``(n, dim)`` de normales estándar según el método de muestreo.

    - ``mc``: pseudoaleatorias.
    - ``antitetica``: la mitad de las trayectorias son el negativo de la otra mitad.
    - ``sobol``: cuasi-Monte Carlo con secuencia de Sobol aleatorizada (Owen),
      transformada por la inversa de la normal.
    """
    if metodo not in METODOS:
        raise ValueError(f"Método desconocido: {metodo}")
    rng = np.random.default_rng(semilla)
    if metodo == "mc":
        return rng.standard_normal((n, dim))
    if metodo == "antitetica":
        mitad = (n + 1) // 2
        z = rng.standard_normal((mitad, dim))
        return np.concatenate([z, -z], axis=0)[:n]
    motor = qmc.Sobol(d=dim, scramble=True, seed=rng)
    m = int(np.ceil(np.log2(max(n, 2))))
    u = motor.random_base2(m)[:n]
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return stats.norm.ppf(u)


@dataclass
class EscenariosMBG:
    """Resultado de una simulación: ``X`` tiene forma ``(n, T+1, Z)``."""

    X: np.ndarray
    anios: np.ndarray
    mu: np.ndarray
    cov: np.ndarray
    metodo: str

    @property
    def n(self) -> int:
        return self.X.shape[0]

    def total(self, pesos: np.ndarray) -> np.ndarray:
        """Índice agregado ponderado (p. ej., por participación en la generación)."""
        return np.einsum("ntz,z->nt", self.X, pesos)


def simular(n: int, anios: np.ndarray, mu: np.ndarray, cov: np.ndarray,
            metodo: str = "mc", semilla: int | None = None,
            intensidad_saltos: float = 0.0, media_salto: float = -0.35,
            desv_salto: float = 0.10, compensar_saltos: bool = True) -> EscenariosMBG:
    """Simula ``n`` trayectorias del MBG multivariado en los años dados.

    El primer año es el de partida (``X = 1``). Los incrementos anuales usan
    la solución exacta; los saltos, si se activan, afectan a todas las zonas
    por igual (choque sistémico, como la pandemia de 2020).
    """
    anios = np.asarray(anios)
    mu = np.asarray(mu, float)
    cov = np.asarray(cov, float)
    z_dim = mu.size
    dt = np.diff(anios).astype(float)
    pasos = dt.size
    L = np.linalg.cholesky(cov)

    deriva = mu - 0.5 * np.diag(cov)
    if intensidad_saltos > 0 and compensar_saltos:
        k = np.exp(media_salto + 0.5 * desv_salto**2) - 1.0
        deriva = deriva - intensidad_saltos * k

    eps = normales(n, pasos * z_dim, metodo, semilla).reshape(n, pasos, z_dim)
    choques = np.einsum("ntk,zk->ntz", eps, L) * np.sqrt(dt)[None, :, None]
    incrementos = deriva[None, None, :] * dt[None, :, None] + choques

    if intensidad_saltos > 0:
        rng = np.random.default_rng(None if semilla is None else semilla + 7919)
        cuentas = rng.poisson(intensidad_saltos * dt, size=(n, pasos))
        tam = (media_salto * cuentas
               + desv_salto * np.sqrt(cuentas) * rng.standard_normal((n, pasos)))
        incrementos = incrementos + tam[:, :, None]

    logx = np.concatenate([np.zeros((n, 1, z_dim)), np.cumsum(incrementos, axis=1)], axis=1)
    return EscenariosMBG(np.exp(logx), anios, mu, cov, metodo)


# ----------------------------------------------------------------------
# Momentos teóricos
# ----------------------------------------------------------------------
def media_teorica(mu: np.ndarray, t: float) -> np.ndarray:
    """E[X(t)] = exp(mu t) con X(0) = 1."""
    return np.exp(np.asarray(mu) * t)


def var_log_teorica(cov: np.ndarray, t: float) -> np.ndarray:
    """Var[ln X(t)] = Sigma_ii t."""
    return np.diag(cov) * t


def cuantiles_teoricos(mu: float, sigma: float, t: float, q: np.ndarray) -> np.ndarray:
    """Cuantiles de la lognormal X(t) = exp((mu - s^2/2) t + s W_t)."""
    return np.exp((mu - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * stats.norm.ppf(q))


# ----------------------------------------------------------------------
# Calibración por máxima verosimilitud con observaciones irregulares
# ----------------------------------------------------------------------
@dataclass
class AjusteMBG:
    mu: float
    sigma: float
    deriva_log: float
    ee_deriva_log: float
    ee_sigma: float
    n_incrementos: int
    log_verosimilitud: float


def calibrar(anios: np.ndarray, valores: np.ndarray) -> AjusteMBG:
    """MLE de (mu, sigma) para un MBG observado en fechas irregulares.

    Con ``y = ln X`` y ``dy_k ~ N(m dt_k, sigma^2 dt_k)`` independientes, los
    estimadores de máxima verosimilitud son

        m_hat = sum dy_k / sum dt_k,
        sigma_hat^2 = (1/n) sum (dy_k - m_hat dt_k)^2 / dt_k,
        mu_hat = m_hat + sigma_hat^2 / 2.
    """
    t = np.asarray(anios, float)
    y = np.log(np.asarray(valores, float))
    dt = np.diff(t)
    dy = np.diff(y)
    if np.any(dt <= 0):
        raise ValueError("Las fechas deben ser estrictamente crecientes.")
    n = dy.size
    m = dy.sum() / dt.sum()
    s2 = np.mean((dy - m * dt) ** 2 / dt)
    ll = -0.5 * np.sum(np.log(2 * np.pi * s2 * dt) + (dy - m * dt) ** 2 / (s2 * dt))
    return AjusteMBG(mu=m + 0.5 * s2, sigma=float(np.sqrt(s2)), deriva_log=float(m),
                     ee_deriva_log=float(np.sqrt(s2 / dt.sum())),
                     ee_sigma=float(np.sqrt(s2 / (2 * n))), n_incrementos=n,
                     log_verosimilitud=float(ll))


def calibrar_por_tramos(anios: np.ndarray, valores: np.ndarray,
                        excluir: np.ndarray) -> AjusteMBG:
    """Calibración excluyendo los incrementos marcados (p. ej., un salto).

    ``excluir`` es un arreglo booleano de longitud ``len(anios) - 1`` que marca
    los incrementos que se descartan. Equivale a tratar esos incrementos como
    saltos y no como difusión.
    """
    t = np.asarray(anios, float)
    y = np.log(np.asarray(valores, float))
    dt = np.diff(t)[~excluir]
    dy = np.diff(y)[~excluir]
    n = dy.size
    m = dy.sum() / dt.sum()
    s2 = np.mean((dy - m * dt) ** 2 / dt)
    ll = -0.5 * np.sum(np.log(2 * np.pi * s2 * dt) + (dy - m * dt) ** 2 / (s2 * dt))
    return AjusteMBG(mu=m + 0.5 * s2, sigma=float(np.sqrt(s2)), deriva_log=float(m),
                     ee_deriva_log=float(np.sqrt(s2 / dt.sum())),
                     ee_sigma=float(np.sqrt(s2 / (2 * n))), n_incrementos=n,
                     log_verosimilitud=float(ll))


def escenarios_desde_parametros(p, n: int, metodo: str = "mc",
                                semilla: int | None = 2026) -> EscenariosMBG:
    """Atajo: simula con los parámetros de demanda del modelo."""
    cov = covarianza_factorial(p.demanda.sigma_comun, p.zonas["sigma_idio"].to_numpy(float))
    d = p.demanda
    return simular(n, p.anios, p.mu_total, cov, metodo, semilla,
                   d.intensidad_saltos, d.media_salto, d.desv_salto, d.compensar_saltos)
