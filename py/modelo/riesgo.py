"""Medidas de riesgo sobre distribuciones simuladas de costo.

Convención: las variables son pérdidas (costo social en valor presente), de
modo que la cola relevante es la superior.

    VaR_a(L)  = cuantil a de L
    CVaR_a(L) = E[L | L >= VaR_a(L)]
              = min_c { c + E[(L - c)^+] / (1 - a) }      (Rockafellar y Uryasev, 2000)
"""

from __future__ import annotations

import numpy as np


def var(perdidas: np.ndarray, alpha: float = 0.95) -> float:
    return float(np.quantile(np.asarray(perdidas, float), alpha))


def cvar(perdidas: np.ndarray, alpha: float = 0.95) -> float:
    """Estimador de Rockafellar-Uryasev evaluado en el VaR empírico."""
    x = np.asarray(perdidas, float)
    v = np.quantile(x, alpha)
    return float(v + np.mean(np.maximum(x - v, 0.0)) / (1.0 - alpha))


def error_estandar(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(x.size))


def ic_bootstrap(x: np.ndarray, estadistico=np.mean, b: int = 1000, nivel: float = 0.95,
                 semilla: int | None = 5) -> tuple[float, float]:
    """Intervalo bootstrap de percentiles para un estadístico de la muestra."""
    rng = np.random.default_rng(semilla)
    x = np.asarray(x, float)
    idx = rng.integers(0, x.size, size=(b, x.size))
    est = np.array([estadistico(x[i]) for i in idx])
    a = (1 - nivel) / 2
    return float(np.quantile(est, a)), float(np.quantile(est, 1 - a))


def frontera_pareto(media: np.ndarray, riesgo: np.ndarray) -> np.ndarray:
    """Índices de los puntos no dominados al minimizar ambas coordenadas."""
    media = np.asarray(media, float)
    riesgo = np.asarray(riesgo, float)
    orden = np.lexsort((riesgo, media))
    frente = []
    mejor = np.inf
    for k in orden:
        if riesgo[k] < mejor - 1e-12:
            frente.append(k)
            mejor = riesgo[k]
    return np.array(frente, dtype=int)


def cvar_normal(mu: float, sigma: float, alpha: float = 0.95) -> float:
    """CVaR analítico de una normal (para validación)."""
    from scipy import stats

    z = stats.norm.ppf(alpha)
    return float(mu + sigma * stats.norm.pdf(z) / (1 - alpha))
