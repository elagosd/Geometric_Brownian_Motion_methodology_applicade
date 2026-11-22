"""Fase I (2013) y comparación del metro elevado y subterráneo.

Uso:  python correr_fase_2013.py [--n 1024]

1. Calibra el modelo con los datos de 2013 y simula la demanda vista desde 2013.
2. Evalúa ex ante (decisión en 2014) el metro elevado, el subterráneo con el
   costo oficial del diseño y con los costos unitarios de 2022, y no hacer
   metro, con y sin política de suelo.
3. Evalúa ex post las mismas alternativas sobre la trayectoria observada
   2013-2025 (censo 2018, pandemia) y los escenarios de la fase II.
4. Calcula el costo urbano anual del viaducto que haría preferible el
   subterráneo y compara tipologías para las líneas 2 y 3 en la fase II.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import rutas
from modelo import gbm
from modelo import metro as M
from modelo.evaluacion import Diseno, evaluar, factores_descuento, muestrear_sobrecostos
from modelo.fases import (ARCHIVO_CAL_2013, TRAYECTORIA_OBSERVADA, cargar_o_calibrar_2013,
                          escenarios_expost)
from modelo.parametros import parametros_calibrados
from modelo.riesgo import cvar, error_estandar

S = rutas.SALIDAS


def guardar_json(nombre, obj):
    (S / nombre).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def resumir(res: dict, base: str) -> pd.DataFrame:
    vb = res[base].vp
    filas = []
    for nom, r in res.items():
        bn = vb - r.vp
        filas.append({"diseno": nom, "vp_media": r.vp.mean(), "vp_cvar": cvar(r.vp), "vp_desv": r.vp.std(ddof=1),
                      "bn_media": bn.mean(), "bn_ee": error_estandar(bn), "prob_bn_pos": float(np.mean(bn > 0)),
                      "capex_medio": float(r.capex_total.mean()),
                      "tiempo_2030": float(r.tiempo_medio[:, np.searchsorted(r.anios, 2030)].mean()),
                      "tiempo_2050": float(r.tiempo_medio[:, -1].mean())})
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    args = ap.parse_args()
    n = args.n
    t0 = time.time()

    p25 = parametros_calibrados()
    p13 = cargar_o_calibrar_2013(p25)
    cal = json.loads(ARCHIVO_CAL_2013.read_text(encoding="utf-8"))
    guardar_json("fase2013_calibracion.json", cal)
    print(f"[1] 2013 calibrado: {cal['indicadores']}")

    esc = gbm.escenarios_desde_parametros(p13, n, "sobol", 2013)
    sc = muestrear_sobrecostos(p13, n)
    tot = esc.total(p13.participacion_generacion)
    obs = np.array([TRAYECTORIA_OBSERVADA[a] for a in range(2013, 2026)])
    pct_obs = [float(np.mean(tot[:, k] <= obs[k])) for k in range(obs.size)]
    np.savez_compressed(S / "fase2013_escenarios.npz", anios=esc.anios, total=tot, observado=obs)
    print(f"[2] Índice 2013->2025: mediana simulada {np.median(tot[:, 12]):.3f}, observado {obs[-1]:.3f} "
          f"(percentil {100 * pct_obs[-1]:.1f})")

    # ---------------------------------------------------------------- ex ante
    disenos = {}
    for th in (0.0, 1.0):
        disenos[f"Sin metro | θ={th:g}"] = Diseno((), th)
        disenos[f"Elevado | θ={th:g}"] = Diseno(("L1E",), th)
        disenos[f"Subterráneo (costo oficial) | θ={th:g}"] = Diseno(("L1S",), th)
        disenos[f"Subterráneo (costo unitarios) | θ={th:g}"] = Diseno(("L1S_alto",), th)
    # Prueba de equidad: el subterráneo con el calendario oficial de 2014 (apertura en 2021, como el elevado).
    disenos["Subterráneo (costo oficial, apertura 2021) | θ=0"] = Diseno(("L1S",), 0.0, (("L1S", 2021),))
    res = {nom: evaluar(p13, d, esc, sc, anio_inicio=2014) for nom, d in disenos.items()}
    df = resumir(res, "Sin metro | θ=0")
    df.to_csv(S / "fase2013_exante.csv", index=False)
    print("[3] Ex ante:\n", df[["diseno", "vp_media", "vp_cvar", "bn_media", "prob_bn_pos"]].round(2).to_string())

    # Tasa de descuento: 12 % (vigente en 2013) frente a 9 % y 6,4 %.
    filas = []
    for tasa in (0.064, 0.09, 0.12):
        q = p13.ajustar("economia", tasa_descuento=tasa)
        base = res["Sin metro | θ=0"]
        d_ = factores_descuento(q, base.anios)
        vb = base.costo_anual @ d_
        for nom, r in res.items():
            v = r.costo_anual @ d_
            filas.append({"tasa": tasa, "diseno": nom, "bn_media": float((vb - v).mean()),
                          "prob_bn_pos": float(np.mean(vb - v > 0))})
    pd.DataFrame(filas).to_csv(S / "fase2013_descuento.csv", index=False)

    # Umbral de externalidad del viaducto (diseños con θ=0).
    umbrales = []
    for tasa in (0.064, 0.09, 0.12):
        q = p13.ajustar("economia", tasa_descuento=tasa)
        d_ = factores_descuento(q, res["Elevado | θ=0"].anios)
        ve = (res["Elevado | θ=0"].costo_anual @ d_).mean()
        for etiqueta in ("Subterráneo (costo oficial) | θ=0", "Subterráneo (costo unitarios) | θ=0",
                         "Subterráneo (costo oficial, apertura 2021) | θ=0"):
            vs = (res[etiqueta].costo_anual @ d_).mean()
            umbrales.append({"tasa": tasa, "alternativa": etiqueta, "diferencia_vp": vs - ve,
                             "externalidad_umbral": M.externalidad_umbral(vs - ve, res[etiqueta].anios, 2021, tasa, 2014)})
    pd.DataFrame(umbrales).to_csv(S / "fase2013_umbral.csv", index=False)
    print("[4] Umbrales:\n", pd.DataFrame(umbrales).round(3).to_string())

    # ---------------------------------------------------------------- ex post
    esc25 = gbm.escenarios_desde_parametros(p25, n, "sobol", 2026)
    esc_ep = escenarios_expost(p13, esc25)
    disenos_ep = {
        "Sin metro": Diseno(()),
        "Elevado (apertura real 2028)": Diseno(("L1E",), 0.0, (("L1E", 2028),)),
        "Elevado (si se hubiera decidido en 2014, 2021)": Diseno(("L1E",), 0.0, (("L1E", 2021),)),
        "Subterráneo oficial (2024)": Diseno(("L1S",), 0.0, (("L1S", 2024),)),
        "Subterráneo oficial (2027)": Diseno(("L1S",), 0.0, (("L1S", 2027),)),
        "Subterráneo oficial (2030)": Diseno(("L1S",), 0.0, (("L1S", 2030),)),
        "Subterráneo unitarios (2027)": Diseno(("L1S_alto",), 0.0, (("L1S_alto", 2027),)),
    }
    res_ep = {nom: evaluar(p13, d, esc_ep, sc, anio_inicio=2014) for nom, d in disenos_ep.items()}
    df_ep = resumir(res_ep, "Sin metro")
    df_ep.to_csv(S / "fase2013_expost.csv", index=False)
    print("[5] Ex post:\n", df_ep[["diseno", "vp_media", "bn_media", "prob_bn_pos"]].round(2).to_string())

    tray = {}
    for nom in ("Sin metro | θ=0", "Elevado | θ=0", "Subterráneo (costo oficial) | θ=0"):
        r = res[nom]
        tray[nom] = {"anios": r.anios.tolist(), "tiempo_p50": np.median(r.tiempo_medio, 0).tolist(),
                     "tiempo_p5": np.percentile(r.tiempo_medio, 5, 0).tolist(),
                     "tiempo_p95": np.percentile(r.tiempo_medio, 95, 0).tolist()}
    guardar_json("fase2013_trayectorias.json", tray)

    # ------------------------------------------- fase II: tipología de L2 y L3
    p = p25
    pr = p.proyectos.copy()
    extra = []
    for pid, km, est in (("L2", 15.5, 11), ("L3", 24.5, 18)):
        c = M.comparar(km, est)
        fila = pr[pr["id"] == pid].iloc[0].copy()
        fila["id"] = pid + "E"
        fila["nombre"] = fila["nombre"] + " — variante elevada"
        fila["capex_billones"] = round(fila["capex_billones"] - c["ahorro_elevado"], 2)
        fila["fuente"] = "Variante elevada: CAPEX subterráneo menos el ahorro por costos unitarios EMB 2023"
        extra.append(fila)
    q = p.con(proyectos=pd.concat([pr, pd.DataFrame(extra)], ignore_index=True))
    esc2 = gbm.escenarios_desde_parametros(q, n, "sobol", 2026)
    sc2 = muestrear_sobrecostos(q, n)
    dis2 = {"Base | θ=0": Diseno(()), "L2 | θ=0": Diseno(("L2",)), "L2E | θ=0": Diseno(("L2E",)),
            "L3 | θ=0": Diseno(("L3",)), "L3E | θ=0": Diseno(("L3E",)),
            "RTN | θ=1": Diseno(("RTN",), 1.0), "L3+RTN | θ=1": Diseno(("L3", "RTN"), 1.0),
            "L3E+RTN | θ=1": Diseno(("L3E", "RTN"), 1.0)}
    res2 = {nom: evaluar(q, d, esc2, sc2) for nom, d in dis2.items()}
    df2 = resumir(res2, "Base | θ=0")
    for sub, ele in (("L2 | θ=0", "L2E | θ=0"), ("L3 | θ=0", "L3E | θ=0"), ("L3+RTN | θ=1", "L3E+RTN | θ=1")):
        dif = res2[sub].vp.mean() - res2[ele].vp.mean()
        pid = sub.split(" ")[0].split("+")[0]
        ap_ = int(q.proyectos.set_index("id").loc[pid, "apertura"])
        df2.loc[df2["diseno"] == ele, "externalidad_umbral"] = M.externalidad_umbral(
            dif, res2[sub].anios, ap_, q.economia.tasa_descuento, q.economia.anio_vp)
    df2.to_csv(S / "fase2_tipologia.csv", index=False)
    print("[6] Tipología fase II:\n", df2[["diseno", "vp_media", "bn_media", "prob_bn_pos"]].round(2).to_string())

    costos = [dict(linea=nom, **M.comparar(km, est)) for nom, km, est in
              (("L1 (trazado elevado, 23,96 km)", 23.96, 16), ("L1 subterránea 2014 (27 km)", 27.0, 28),
               ("L2 (15,5 km)", 15.5, 11), ("L3 (24,5 km)", 24.5, 18))]
    pd.DataFrame(costos).to_csv(S / "metro_costos.csv", index=False)
    guardar_json("fase2013_resumen.json", {
        "n": n, "segundos": time.time() - t0, "percentil_observado_2025": pct_obs[-1],
        "percentiles_observados": pct_obs, "otros_costos_l1": M.otros_costos_l1(),
        "capex": {k: M.capex_total_2025(k) for k in M.CAPEX_2022}, "indexacion": M.INDEXACION_2022_2025})
    print(f"Listo en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
