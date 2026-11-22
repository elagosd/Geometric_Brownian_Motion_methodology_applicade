"""Pruebas del modelo: propiedades matemáticas y coherencia numérica.

Ejecutar desde la raíz del proyecto:  python -m pytest -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "py"))

from modelo import gbm, riesgo  # noqa: E402
from modelo.asignacion import equilibrio, funcion_congestion  # noqa: E402
from modelo.demanda import furness, matriz_od  # noqa: E402
from modelo.evaluacion import Diseno, evaluar, muestrear_sobrecostos  # noqa: E402
from modelo.opciones_reales import lsm_mcdonald_siegel, mcdonald_siegel  # noqa: E402
from modelo.parametros import parametros_calibrados  # noqa: E402
from modelo.red import abiertos_en, construir_oferta  # noqa: E402


@pytest.fixture(scope="module")
def p():
    return parametros_calibrados()


# --------------------------------------------------------------------- MBG
def test_mbg_media_y_varianza():
    mu = np.array([0.01, 0.03, -0.01])
    cov = gbm.covarianza_factorial(0.05, np.array([0.03, 0.04, 0.02]))
    anios = np.arange(2025, 2046)
    esc = gbm.simular(40000, anios, mu, cov, metodo="antitetica", semilla=1)
    t = anios[-1] - anios[0]
    media = esc.X[:, -1, :].mean(0)
    np.testing.assert_allclose(media, gbm.media_teorica(mu, t), rtol=0.01)
    var_log = np.log(esc.X[:, -1, :]).var(0)
    np.testing.assert_allclose(var_log, gbm.var_log_teorica(cov, t), rtol=0.03)


def test_mbg_correlacion():
    cov = gbm.covarianza_factorial(0.05, np.full(4, 0.03))
    esc = gbm.simular(30000, np.arange(2025, 2027), np.zeros(4), cov, semilla=2)
    r = np.corrcoef(np.log(esc.X[:, 1, :]).T)
    np.testing.assert_allclose(r, gbm.correlacion(cov), atol=0.02)


def test_sobol_y_antiteticas_reducen_error():
    mu, cov = np.array([0.02]), np.array([[0.04]])
    anios = np.arange(2025, 2036)
    exacto = gbm.media_teorica(mu, 10)[0]
    err = {}
    for metodo in ("mc", "antitetica", "sobol"):
        est = [gbm.simular(512, anios, mu, cov, metodo, semilla=s).X[:, -1, 0].mean() for s in range(30)]
        err[metodo] = np.sqrt(np.mean((np.array(est) - exacto) ** 2))
    assert err["antitetica"] < err["mc"]
    assert err["sobol"] < err["mc"]


def test_calibracion_mle_recupera_parametros():
    rng = np.random.default_rng(4)
    t = np.cumsum(rng.uniform(0.5, 2.0, 400))
    mu, s = 0.03, 0.08
    dt = np.diff(t, prepend=0.0)
    y = np.cumsum((mu - 0.5 * s**2) * dt + s * np.sqrt(dt) * rng.standard_normal(t.size))
    aj = gbm.calibrar(t, np.exp(y))
    assert abs(aj.sigma - s) < 4 * aj.ee_sigma
    assert abs(aj.deriva_log - (mu - 0.5 * s**2)) < 4 * aj.ee_deriva_log


def test_saltos_compensados_conservan_la_media():
    mu, cov = np.array([0.02]), np.array([[0.0025]])
    anios = np.arange(2025, 2046)
    esc = gbm.simular(60000, anios, mu, cov, semilla=5, intensidad_saltos=0.04,
                      media_salto=-0.35, desv_salto=0.1, compensar_saltos=True)
    np.testing.assert_allclose(esc.X[:, -1, 0].mean(), np.exp(0.02 * 20), rtol=0.02)


# ------------------------------------------------------------------ riesgo
def test_cvar_normal():
    rng = np.random.default_rng(6)
    x = rng.normal(10, 2, 400000)
    assert riesgo.cvar(x, 0.95) == pytest.approx(riesgo.cvar_normal(10, 2, 0.95), rel=5e-3)
    assert riesgo.cvar(x) >= riesgo.var(x) >= x.mean()


def test_frontera_pareto():
    m = np.array([1, 2, 3, 1.5, 2.5])
    c = np.array([5, 3, 1, 6, 3.5])
    assert set(riesgo.frontera_pareto(m, c)) == {0, 1, 2}


# --------------------------------------------------------- red y demanda
def test_furness_margenes():
    rng = np.random.default_rng(7)
    P = rng.uniform(1, 5, (3, 6))
    A = rng.uniform(1, 5, (3, 6))
    A = A / A.sum(1, keepdims=True) * P.sum(1, keepdims=True)
    f = np.exp(-rng.uniform(0, 2, (6, 6)))
    T = furness(P, A, f, iteraciones=200)
    np.testing.assert_allclose(T.sum(2), P, rtol=1e-6)
    np.testing.assert_allclose(T.sum(1), A, rtol=1e-6)


def test_funcion_congestion_monotona():
    x = np.linspace(0, 5, 501)
    f = funcion_congestion(x, 0.15, 4, 2.0, 10.0)
    assert np.all(np.diff(f) >= -1e-12)
    assert f[0] == pytest.approx(1.0)
    assert f.max() <= 10.0 + 1e-12


def test_incidencia_caminos(p):
    o = construir_oferta(p, abiertos_en(p, 2025, ()))
    Z = o.Z
    # Todo par distinto tiene camino y los intrazonales no usan arcos.
    n_arcos = o.A_via.sum(0).reshape(Z, Z)
    assert np.all(np.diag(n_arcos) == 0)
    assert np.all(n_arcos[~np.eye(Z, dtype=bool)] >= 1)


def test_calibracion_reproduce_metas(p):
    from modelo.calibracion import indicadores_base

    ind = indicadores_base(p)
    assert ind["vel_via"] == pytest.approx(p.metas.vel_via, rel=1e-3)
    assert ind["part_tp"] == pytest.approx(p.metas.part_tp, rel=1e-3)
    assert ind["tiempo_medio"] == pytest.approx(p.metas.tiempo_medio, rel=1e-3)


def test_equilibrio_unico_y_monotono(p):
    o = construir_oferta(p, abiertos_en(p, 2025, ()))
    tiempos = []
    for x in (1.0, 1.05, 1.10):
        T = matriz_od(p, o, np.full((1, p.n_zonas), x), 2025, 0.0)
        a = equilibrio(p.ajustar("red", tol_equilibrio=1e-9), o, T, iteraciones=800)
        b = equilibrio(p.ajustar("red", tol_equilibrio=1e-9), o, T,
                       s0=np.full_like(a.s_tp, 0.2), iteraciones=800)
        assert a.tiempo_medio[0] == pytest.approx(b.tiempo_medio[0], rel=1e-4)
        tiempos.append(a.tiempo_medio[0])
    assert tiempos[0] < tiempos[1] < tiempos[2]


def test_evaluacion_numeros_comunes(p):
    esc = gbm.escenarios_desde_parametros(p, 64, semilla=8)
    sc = muestrear_sobrecostos(p, 64)
    r1 = evaluar(p, Diseno(), esc, sc, paso=5)
    r2 = evaluar(p, Diseno(), esc, sc, paso=5)
    np.testing.assert_allclose(r1.vp, r2.vp)
    r3 = evaluar(p, Diseno(("RTN",)), esc, sc, paso=5)
    assert r3.vp_componente("capex").mean() > 0
    assert np.all(r3.componentes["residual"][:, -1] <= 0)


# ------------------------------------------------------- opciones reales
def test_lsm_contra_mcdonald_siegel():
    analitico, _, _ = mcdonald_siegel(1.0, 1.0, 0.02, 0.2, 0.06)
    estimado, ee = lsm_mcdonald_siegel(1.0, 1.0, 0.02, 0.2, 0.06, horizonte=100, dt=0.05, n=20000)
    assert estimado == pytest.approx(analitico, rel=0.05)


# ---------------------------------------------------------------- fase I y metro
def test_metro_prima_y_umbral():
    from modelo import metro
    c = metro.comparar(23.96, 16)
    assert c["capex_subterraneo"] > c["capex_elevado"] > 0
    assert 0.5 < c["prima"] < 1.0
    # La L1 elevada con unitarios reproduce su CAPEX total (los otros costos cierran la diferencia).
    assert abs(metro.costo_linea("elevado", 23.96, 16).capex - metro.capex_total_2025("elevado_factibilidad_2017")) < 1e-9
    # Umbral: diferencia / suma de factores de descuento de los años de operación.
    anios = np.arange(2014, 2051)
    u = metro.externalidad_umbral(1.0, anios, 2021, 0.12, 2014)
    suma = sum(1.12 ** -(a - 2014) for a in range(2021, 2051))
    assert np.isclose(u, 1.0 / suma)


def test_fase2013_parametros():
    from modelo import fases
    p13 = fases.parametros_2013(parametros_calibrados())
    assert p13.demanda.anio_base == 2013 and p13.economia.tasa_descuento == 0.12
    assert set(p13.proyectos["id"]) == {"L1E", "L1S", "L1S_alto"}
    bog = (p13.zonas["ambito"] == "Bogotá").to_numpy()
    assert np.isclose(p13.zonas.loc[bog, "poblacion_miles"].sum(), fases.POB_BOGOTA_2013, rtol=0.01)
