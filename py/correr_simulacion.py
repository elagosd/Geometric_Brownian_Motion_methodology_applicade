"""Corrida completa del estudio: calibración, Monte Carlo, optimización,
sensibilidad, saltos, convergencia y opciones reales.

Uso:  python correr_simulacion.py [--n 1000] [--rapido]

Escribe en ``salidas/`` los archivos que leen los scripts de figuras y los
generadores de cifras y tablas del documento.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import rutas
from modelo import gbm
from modelo.calibracion import ARCHIVO as ARCHIVO_CAL
from modelo.evaluacion import Diseno, evaluar, factores_descuento, muestrear_sobrecostos
from modelo.opciones_reales import (beneficios_potenciales, lsm, lsm_mcdonald_siegel,
                                    mcdonald_siegel)
from modelo.optimizacion import elegir, enumerar_disenos, evaluar_todos, resumen, ruta_optima
from modelo.parametros import cargar_serie_transmilenio, parametros_calibrados
from modelo.riesgo import cvar

S = rutas.SALIDAS
SEMILLA = 2026
BASE = "Base | θ=0"


def guardar_json(nombre, obj):
    (S / nombre).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def percentiles(a: np.ndarray, qs=(5, 25, 50, 75, 95)) -> dict:
    return {f"p{q}": np.percentile(a, q, axis=0).tolist() for q in qs} | {"media": a.mean(0).tolist()}


def paso_calibracion_mbg():
    serie = cargar_serie_transmilenio()
    anios = serie["anio"].to_numpy()
    valores = serie["pasajeros_dia_millones"].to_numpy()
    completo = gbm.calibrar(anios, valores)
    # Choque (2019->2020) y rebote (2020->2023) se tratan como un salto y no como difusión.
    excluir = ((anios[:-1] == 2019) & (anios[1:] == 2020)) | ((anios[:-1] == 2020) & (anios[1:] == 2023))
    robusto = gbm.calibrar_por_tramos(anios, valores, excluir)
    salto = float(np.log(valores[anios == 2020][0] / valores[anios == 2019][0]))
    out = {"completo": completo.__dict__, "sin_pandemia": robusto.__dict__, "salto_2020_log": salto,
           "anios": anios.tolist(), "valores": valores.tolist()}
    guardar_json("calibracion_mbg.json", out)
    return out


def paso_convergencia(p, reps: int, tamanos):
    """RMSE del VP esperado del diseño base según método y tamaño de muestra."""
    sc_ref = muestrear_sobrecostos(p, 8192, semilla=11)
    ref = evaluar(p, Diseno(), gbm.escenarios_desde_parametros(p, 8192, "sobol", 99),
                  sc_ref, paso=5).vp.mean()
    filas = []
    for metodo in ("mc", "antitetica", "sobol"):
        for n in tamanos:
            est = []
            for r in range(reps):
                esc = gbm.escenarios_desde_parametros(p, n, metodo, semilla=1000 + r)
                sc = muestrear_sobrecostos(p, n, semilla=11)
                est.append(evaluar(p, Diseno(), esc, sc, paso=5).vp.mean())
            est = np.array(est)
            filas.append({"metodo": metodo, "n": n, "rmse": float(np.sqrt(np.mean((est - ref) ** 2))),
                          "sesgo": float(est.mean() - ref), "desv": float(est.std(ddof=1))})
    df = pd.DataFrame(filas)
    df.to_csv(S / "convergencia.csv", index=False)
    guardar_json("convergencia_ref.json", {"referencia": float(ref)})
    return df


def paso_sensibilidad(p, d_opt: Diseno, n: int):
    """Beneficio neto esperado del diseño óptimo frente a la base, un factor a la vez."""
    casos = {
        "Valor del tiempo (VTTS)": ("economia", "vtts_cop_h", 5000.0, 12000.0),
        "Tasa social de descuento": ("economia", "tasa_descuento", 0.064, 0.12),
        "Volatilidad común": ("demanda", "sigma_comun", 0.025, 0.10),
        "Tendencia común de viajes": ("demanda", "mu_comun", 0.0, 0.01),
        "Horas pico equivalentes/año": ("economia", "factor_anual", 2400.0, 3600.0),
        "Costo de la política de suelo": ("suelo", "costo_anual_billones", 0.3, 0.9),
        "Viajes que pasan a no motorizados": ("suelo", "reduccion_motorizada_max", 0.025, 0.075),
        "Relocalización del empleo central": ("suelo", "relocalizacion_centro_max", 0.075, 0.225),
        "Sobrecostos de CAPEX": ("sobrecostos", "activar", False, True),
    }
    esc = gbm.escenarios_desde_parametros(p, n, semilla=SEMILLA)
    sc = muestrear_sobrecostos(p, n)

    def bn(q, esc_q):
        sc_q = muestrear_sobrecostos(q, n)
        return float((evaluar(q, Diseno(), esc_q, sc_q).vp - evaluar(q, d_opt, esc_q, sc_q).vp).mean())

    central = float((evaluar(p, Diseno(), esc, sc).vp - evaluar(p, d_opt, esc, sc).vp).mean())
    filas = []
    for etiqueta, (bloque, campo, bajo, alto) in casos.items():
        vals = []
        for v in (bajo, alto):
            q = p.ajustar(bloque, **{campo: v})
            esc_q = gbm.escenarios_desde_parametros(q, n, semilla=SEMILLA) if bloque == "demanda" else esc
            vals.append(bn(q, esc_q))
        filas.append({"factor": etiqueta, "bajo": bajo, "alto": alto, "bn_bajo": vals[0], "bn_alto": vals[1]})
    df = pd.DataFrame(filas)
    df["rango"] = (df["bn_alto"] - df["bn_bajo"]).abs()
    df.sort_values("rango", ascending=False).to_csv(S / "sensibilidad.csv", index=False)
    guardar_json("sensibilidad_central.json", {"bn_central": central, "diseno": d_opt.nombre, "n": n})
    return df


def paso_descuento(p, resultados, nombres):
    filas = []
    for r_desc in (0.035, 0.064, 0.09, 0.12):
        q = p.ajustar("economia", tasa_descuento=r_desc)
        base = resultados[BASE]
        desc = factores_descuento(q, base.anios)
        vb = base.costo_anual @ desc
        for nom in nombres:
            v = resultados[nom].costo_anual @ desc
            filas.append({"tasa": r_desc, "diseno": nom, "bn_media": float((vb - v).mean()),
                          "prob_bn_pos": float(np.mean(vb - v > 0))})
    df = pd.DataFrame(filas)
    df.to_csv(S / "descuento.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--rapido", action="store_true", help="Corrida de prueba con pocos escenarios")
    args = ap.parse_args()
    n = 128 if args.rapido else args.n
    t0 = time.time()

    p = parametros_calibrados()
    cal = json.loads(ARCHIVO_CAL.read_text(encoding="utf-8"))
    guardar_json("calibracion_red.json", cal)
    print(f"[1] Año base calibrado: {cal['indicadores']}")

    mbg = paso_calibracion_mbg()
    print(f"[2] MBG TransMilenio: sigma completo={mbg['completo']['sigma']:.3f}, "
          f"sin pandemia={mbg['sin_pandemia']['sigma']:.3f}")

    esc = gbm.escenarios_desde_parametros(p, n, "sobol", SEMILLA)
    sc = muestrear_sobrecostos(p, n)
    pesos = p.participacion_generacion
    tot = esc.total(pesos)
    np.savez_compressed(S / "escenarios.npz", anios=esc.anios, total=tot,
                        X_final=esc.X[:, -1, :], muestra=esc.X[:60])
    print(f"[3] {n} escenarios Sobol; índice total 2050: media={tot[:, -1].mean():.3f}")

    disenos = enumerar_disenos(p)
    t = time.time()
    resultados = evaluar_todos(p, disenos, esc, sc, paso=1,
                               progreso=lambda k, m, nom: print(f"    {k}/{m} {nom}", flush=True) if k % 8 == 0 else None)
    print(f"[4] {len(disenos)} diseños evaluados en {time.time() - t:.0f} s")
    df = resumen(resultados, BASE)
    df.to_csv(S / "resumen_disenos.csv", index=False)
    vp = np.stack([resultados[d].vp for d in df["diseno"]])
    np.savez_compressed(S / "vp_disenos.npz", nombres=df["diseno"].to_numpy(), vp=vp)

    ruta = ruta_optima(df)
    ruta.to_csv(S / "ruta_optima.csv", index=False)
    opt0 = elegir(df, 0.0)
    opt5 = elegir(df, 0.5)
    print(f"[5] Óptimo neutral: {opt0['diseno']}; con w=0.5: {opt5['diseno']}")

    nom_opt = opt0["diseno"]
    d_opt = resultados[nom_opt].diseno
    todo = "L1X+L2+L3+RTN | θ=1"
    sin_l3 = Diseno(tuple(x for x in d_opt.proyectos if x != "L3"), d_opt.theta)
    con_l3 = Diseno(tuple(sorted(set(sin_l3.proyectos) | {"L3"})), d_opt.theta)
    clave = sorted({BASE, nom_opt, todo, con_l3.nombre, "L3 | θ=0", "RTN | θ=0", "Base | θ=1"})
    tray = {}
    for nom in clave:
        r = resultados[nom]
        tray[nom] = {"anios": r.anios.tolist(), "tiempo": percentiles(r.tiempo_medio),
                     "vel": percentiles(r.vel_via), "tp": percentiles(r.part_tp),
                     "hac": percentiles(r.hac_max), "co2": percentiles(r.co2_mt),
                     "componentes_vp": {k: float(r.vp_componente(k).mean()) for k in r.componentes}}
    guardar_json("trayectorias.json", tray)

    paso_descuento(p, resultados, [nom_opt, con_l3.nombre, "L2 | θ=0", "L3 | θ=0", todo, "RTN | θ=0", "Base | θ=1"])
    print("[6] Sensibilidad a la tasa de descuento lista")

    # Saltos sistémicos (choque tipo pandemia, compensado en media).
    q = p.ajustar("demanda", intensidad_saltos=0.04, media_salto=-0.35, desv_salto=0.10)
    esc_j = gbm.escenarios_desde_parametros(q, n, "sobol", SEMILLA)
    filas = []
    for nom in clave:
        d = resultados[nom].diseno
        vj = evaluar(q, d, esc_j, sc).vp
        v0 = resultados[nom].vp
        filas.append({"diseno": nom, "media_mbg": v0.mean(), "cvar_mbg": cvar(v0), "desv_mbg": v0.std(),
                      "media_saltos": vj.mean(), "cvar_saltos": cvar(vj), "desv_saltos": vj.std()})
    pd.DataFrame(filas).to_csv(S / "saltos.csv", index=False)
    print("[7] Escenario con saltos listo")

    # Opciones reales sobre la Línea 3, manteniendo el resto del diseño óptimo.
    anios_b, B = beneficios_potenciales(p, con_l3, "L3", esc, sc)
    op = lsm(p, "L3", anios_b, B, sc["L3"], np.arange(2027, 2041))
    F_an, v_star, beta = mcdonald_siegel(1.0, 1.0, 0.02, 0.2, 0.06)
    F_lsm, F_ee = lsm_mcdonald_siegel(1.0, 1.0, 0.02, 0.2, 0.06, horizonte=100, dt=0.05, n=20000)
    guardar_json("opciones_reales.json", {
        "diseno_referencia": sin_l3.nombre, "anios_decision": op.anios_decision.tolist(),
        "vpn_por_anio": op.vpn_por_anio.tolist(), "valor_flexible": op.valor_flexible,
        "valor_flexible_ee": op.valor_flexible_ee, "valor_ahora": op.valor_ahora,
        "mejor_anio_estatico": op.mejor_anio_estatico, "valor_estatico": op.valor_estatico,
        "valor_opcion": op.valor_opcion, "prob_nunca": op.prob_nunca,
        "frontera": [None if np.isnan(x) else float(x) for x in op.frontera],
        "ejercicio_anios": np.unique(op.anio_ejercicio, return_counts=True)[0].tolist(),
        "ejercicio_cuentas": np.unique(op.anio_ejercicio, return_counts=True)[1].tolist(),
        "beneficio_anios": anios_b.tolist(), "beneficio": percentiles(B),
        "validacion": {"analitico": F_an, "umbral": v_star, "beta": beta, "lsm": F_lsm, "lsm_ee": F_ee},
    })
    print(f"[8] Opciones reales: flexible={op.valor_flexible:.2f}, ahora={op.valor_ahora:.2f}, "
          f"P(nunca)={op.prob_nunca:.2f}; validación {F_lsm:.4f} vs {F_an:.4f}")

    n_s = 128 if args.rapido else 400
    paso_sensibilidad(p, d_opt, n_s)
    print("[9] Tornado listo")

    reps = 4 if args.rapido else 20
    tam = (32, 64, 128) if args.rapido else (32, 64, 128, 256, 512, 1024)
    paso_convergencia(p, reps, tam)
    print("[10] Convergencia lista")

    guardar_json("corrida.json", {"n": n, "semilla": SEMILLA, "segundos": time.time() - t0,
                                  "optimo_neutral": nom_opt, "optimo_w05": opt5["diseno"]})
    print(f"Listo en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
