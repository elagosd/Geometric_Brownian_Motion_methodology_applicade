"""Genera tex/datos_numericos.tex con las cifras que cita el documento.

Todas las cifras del texto salen de aquí; si cambian las entradas o el
código, basta con volver a correr la simulación y este script.
"""

import json

import numpy as np
import pandas as pd

import rutas
from modelo.parametros import parametros_calibrados

S = rutas.SALIDAS
BASE = "Base | θ=0"


def num(x, dec=1):
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, dec=0):
    return num(100 * x, dec) + r"\,\%"


def tex_nombre(n):
    import re
    n = re.sub(r"(\d)\.(\d)", r"\1{,}\2", n)
    return n.replace("θ", r"$\theta$").replace("|", r"\textbar{}").replace("+", "+\\allowbreak{}")


def umbral_saltos(saltos):
    """Menor peso del CVaR desde el que cambia el diseño elegido entre los evaluados con saltos."""
    primero = None
    for w in np.linspace(0, 1, 1001):
        elegido = ((1 - w) * saltos["media_saltos"] + w * saltos["cvar_saltos"]).idxmin()
        if primero is None:
            primero = elegido
        elif elegido != primero:
            return w
    return np.nan


def macros_fase2013():
    """Cifras de la fase I (2013) y de la comparación elevado vs. subterráneo."""
    if not (S / "fase2013_resumen.json").exists():
        return {}
    from modelo import fases
    res = json.loads((S / "fase2013_resumen.json").read_text(encoding="utf-8"))
    cal = json.loads((S / "fase2013_calibracion.json").read_text(encoding="utf-8"))
    ex = pd.read_csv(S / "fase2013_exante.csv").set_index("diseno")
    xp = pd.read_csv(S / "fase2013_expost.csv").set_index("diseno")
    de = pd.read_csv(S / "fase2013_descuento.csv")
    um = pd.read_csv(S / "fase2013_umbral.csv")
    ti = pd.read_csv(S / "fase2_tipologia.csv").set_index("diseno")
    mc = pd.read_csv(S / "metro_costos.csv").set_index("linea")

    def bn(tasa, nombre):
        return de[(de["tasa"] == tasa) & (de["diseno"] == nombre)].iloc[0]

    def umbral(tasa, alt):
        return um[(um["tasa"] == tasa) & (um["alternativa"] == alt)]["externalidad_umbral"].iloc[0]

    E, SO, SU = "Elevado | θ=0", "Subterráneo (costo oficial) | θ=0", "Subterráneo (costo unitarios) | θ=0"
    SO21 = "Subterráneo (costo oficial, apertura 2021) | θ=0"
    l1e = mc.loc["L1 (trazado elevado, 23,96 km)"]
    l1s = mc.loc["L1 subterránea 2014 (27 km)"]
    l2, l3 = mc.loc["L2 (15,5 km)"], mc.loc["L3 (24,5 km)"]
    obs = res["percentiles_observados"]
    m = {
        "PobBogotaDosMilTrece": num(fases.POB_BOGOTA_2013 / 1000, 2),
        "ViajesPicoDosMilTrece": num(fases.VIAJES_PICO_2013 / 1000, 0),
        "VelDosMilTrece": num(cal["indicadores"]["vel_via"], 1),
        "PartTPDosMilTrece": pct(cal["indicadores"]["part_tp"], 1),
        "TiempoDosMilTrece": num(cal["indicadores"]["tiempo_medio"], 1),
        "HacDosMilTrece": num(cal["indicadores"]["hacinamiento_max"], 2),
        "EscalaDosMilTrece": num(cal["escala_cap_via"], 0),
        "ASCDosMilTrece": num(cal["asc_tp"], 2),
        "PercentilObsFin": pct(res["percentil_observado_2025"], 1),
        "PercentilObsPrepandemia": pct(obs[6], 0),
        "Indexacion": num(res["indexacion"], 3),
        "OtrosCostosLuno": num(res["otros_costos_l1"], 1),
        "CapexElevadoLuno": num(res["capex"]["elevado_factibilidad_2017"], 1),
        "CapexConcesionLuno": num(res["capex"]["elevado_concesion_2019"], 1),
        "CapexSubOficial": num(res["capex"]["subterraneo_2015"], 1),
        "CapexSubUnitarios": num(l1s["capex_subterraneo"], 1),
        "CapexLdosOficial": num(res["capex"]["l2_subterraneo_2021"], 1),
        "PrimaLuno": pct(l1e["prima"], 0),
        "PrimaLdos": pct(l2["prima"], 0),
        "PrimaLtres": pct(l3["prima"], 0),
        "AhorroLuno": num(l1e["ahorro_elevado"], 1),
        "AhorroLdos": num(l2["ahorro_elevado"], 1),
        "AhorroLtres": num(l3["ahorro_elevado"], 1),
        "CapexElevLdos": num(l2["capex_elevado"], 1),
        "CapexSubLdos": num(l2["capex_subterraneo"], 1),
        "CapexElevLtres": num(l3["capex_elevado"], 1),
        "CapexSubLtres": num(l3["capex_subterraneo"], 1),
        "KmPorKm": num(1 / (1 - l1e["ahorro_elevado"] / l1e["capex_subterraneo"]), 2),
    }
    for tasa, suf in ((0.064, "Seis"), (0.09, "Nueve"), (0.12, "Doce")):
        for nombre, et in ((E, "Elev"), (SO, "SubOf"), (SU, "SubUn")):
            f = bn(tasa, nombre)
            m[f"BN{et}{suf}"] = num(f["bn_media"])
            m[f"PBN{et}{suf}"] = pct(f["prob_bn_pos"])
        f = bn(tasa, SO21)
        m[f"BNSubOfVeintiuno{suf}"] = num(f["bn_media"])
        m[f"UmbralOfVeintiuno{suf}"] = num(umbral(tasa, SO21), 2)
        m[f"UmbralOf{suf}"] = num(umbral(tasa, SO), 2)
        m[f"UmbralUn{suf}"] = num(umbral(tasa, SU), 1)
    # Tasa de descuento a la que el subterráneo de costo oficial empata con el elevado (interpolación lineal).
    for nombre, et in ((SO, "TasaEmpate"), (SO21, "TasaEmpateVeintiuno")):
        g = de[de["diseno"].isin([E, nombre])].pivot(index="tasa", columns="diseno", values="bn_media")
        dif = (g[nombre] - g[E]).sort_index()
        t, d = dif.index.to_numpy(), dif.to_numpy()
        k = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
        m[et] = pct(t[k[0]] - d[k[0]] * (t[k[0] + 1] - t[k[0]]) / (d[k[0] + 1] - d[k[0]]), 1) if k.size else "---"
    m["BNSinMetroTheta"] = num(ex.loc["Sin metro | θ=1", "bn_media"])
    m["BNElevTheta"] = num(ex.loc["Elevado | θ=1", "bn_media"])
    m["BNSubOfTheta"] = num(ex.loc["Subterráneo (costo oficial) | θ=1", "bn_media"])
    m["TiempoSinMetroExante"] = num(ex.loc["Sin metro | θ=0", "tiempo_2050"])
    m["TiempoSinMetroExpost"] = num(xp.loc["Sin metro", "tiempo_2050"])
    for nombre, et in (("Elevado (apertura real 2028)", "ElevReal"),
                       ("Elevado (si se hubiera decidido en 2014, 2021)", "ElevDosMilVeintiuno"),
                       ("Subterráneo oficial (2024)", "SubDosMilVeinticuatro"),
                       ("Subterráneo oficial (2027)", "SubDosMilVeintisiete"),
                       ("Subterráneo oficial (2030)", "SubDosMilTreinta"),
                       ("Subterráneo unitarios (2027)", "SubUnDosMilVeintisiete")):
        m[f"BNExpost{et}"] = num(xp.loc[nombre, "bn_media"])
        m[f"PBNExpost{et}"] = pct(xp.loc[nombre, "prob_bn_pos"])
    for d, et in (("L2 | θ=0", "Ldos"), ("L2E | θ=0", "LdosE"), ("L3 | θ=0", "Ltres"), ("L3E | θ=0", "LtresE"),
                  ("RTN | θ=1", "RTNTheta"), ("L3+RTN | θ=1", "LtresRTN"), ("L3E+RTN | θ=1", "LtresERTN")):
        m[f"BNTipo{et}"] = num(ti.loc[d, "bn_media"])
        m[f"PBNTipo{et}"] = pct(ti.loc[d, "prob_bn_pos"])
    m["UmbralTipoLdos"] = num(ti.loc["L2E | θ=0", "externalidad_umbral"], 1)
    m["UmbralTipoLtres"] = num(ti.loc["L3E | θ=0", "externalidad_umbral"], 1)
    m["NFaseUno"] = f"{res['n']:,}".replace(",", ".")
    return m


def main():
    p = parametros_calibrados()
    cal = json.loads((S / "calibracion_red.json").read_text(encoding="utf-8"))
    mbg = json.loads((S / "calibracion_mbg.json").read_text(encoding="utf-8"))
    corr = json.loads((S / "corrida.json").read_text(encoding="utf-8"))
    df = pd.read_csv(S / "resumen_disenos.csv").set_index("diseno")
    op = json.loads((S / "opciones_reales.json").read_text(encoding="utf-8"))
    conv = pd.read_csv(S / "convergencia.csv")
    saltos = pd.read_csv(S / "saltos.csv").set_index("diseno")
    desc = pd.read_csv(S / "descuento.csv")
    sens = json.loads((S / "sensibilidad_central.json").read_text(encoding="utf-8"))
    tray = json.loads((S / "trayectorias.json").read_text(encoding="utf-8"))
    ruta = pd.read_csv(S / "ruta_optima.csv")

    opt = corr["optimo_neutral"]
    o, b = df.loc[opt], df.loc[BASE]
    l3 = df.loc["L3 | θ=0"]
    l2 = df.loc["L2 | θ=0"]
    todo = df.loc["L1X+L2+L3+RTN | θ=1"]
    rtn = df.loc["RTN | θ=0"]
    th1 = df.loc["Base | θ=1"]

    pendientes = {m: np.polyfit(np.log(g["n"]), np.log(g["rmse"]), 1)[0] for m, g in conv.groupby("metodo")}
    n_max = conv["n"].max()
    rmse = conv[conv["n"] == n_max].set_index("metodo")["rmse"]

    def tasa_cero(nombre):
        g = desc[desc["diseno"] == nombre].sort_values("tasa")
        pos = g[g["bn_media"] > 0]
        return pos["tasa"].max() if not pos.empty else np.nan

    l3_desc = desc[desc["diseno"] == "L3 | θ=0"].set_index("tasa")["bn_media"]
    w_cambio = ruta[ruta["diseno"] != ruta["diseno"].iloc[0]]
    tb, to = tray[BASE], tray[opt]

    m = {
        "NEscenarios": f"{corr['n']:,}".replace(",", "."),
        "NDisenos": str(len(df)),
        "SemillaSimulacion": str(corr["semilla"]),
        "ViajesPicoBase": num(p.demanda.viajes_pico_base / 1000, 0),
        "EscalaCapVia": num(cal["escala_cap_via"], 0),
        "ASCTP": num(cal["asc_tp"], 2),
        "BetaGravedad": num(cal["beta_gravedad"], 4),
        "HacinamientoBase": num(cal["indicadores"]["hacinamiento_max"], 2),
        "SigmaTMCompleto": pct(mbg["completo"]["sigma"], 1),
        "SigmaTMRobusto": pct(mbg["sin_pandemia"]["sigma"], 1),
        "MuTMRobusto": pct(mbg["sin_pandemia"]["mu"], 1),
        "SaltoTM": pct(np.exp(mbg["salto_2020_log"]) - 1, 0),
        "SigmaZona": pct(np.sqrt(p.demanda.sigma_comun**2 + 0.03**2), 1),
        "OptimoNeutral": tex_nombre(opt),
        "OptimoWMedio": tex_nombre(corr["optimo_w05"]),
        "VPBase": num(b["vp_media"]),
        "VPBaseEE": num(b["vp_ee"], 2),
        "VPOptimo": num(o["vp_media"]),
        "CVaRBase": num(b["vp_cvar"]),
        "CVaROptimo": num(o["vp_cvar"]),
        "BNOptimo": num(o["bn_media"]),
        "BNOptimoEE": num(o["bn_ee"], 2),
        "PBNOptimo": pct(o["prob_bn_pos"]),
        "BNThetaUno": num(th1["bn_media"]),
        "BNRTN": num(rtn["bn_media"]),
        "PBNRTN": pct(rtn["prob_bn_pos"]),
        "BNLtres": num(l3["bn_media"]),
        "PBNLtres": pct(l3["prob_bn_pos"]),
        "BNLdos": num(l2["bn_media"]),
        "PBNLdos": pct(l2["prob_bn_pos"]),
        "BNTodo": num(todo["bn_media"]),
        "PBNTodo": pct(todo["prob_bn_pos"]),
        "DesvBase": num(b["vp_desv"]),
        "DesvLtres": num(l3["vp_desv"]),
        "DesvTodo": num(todo["vp_desv"]),
        "TiempoBaseFin": num(tb["tiempo"]["media"][-1]),
        "TiempoOptFin": num(to["tiempo"]["media"][-1]),
        "TiempoTodoFin": num(tray["L1X+L2+L3+RTN | θ=1"]["tiempo"]["media"][-1]),
        "VelBaseFin": num(tb["vel"]["media"][-1]),
        "VelOptFin": num(to["vel"]["media"][-1]),
        "TiempoBasePNovFin": num(tb["tiempo"]["p95"][-1]),
        "PendienteMC": num(pendientes.get("mc", np.nan), 2),
        "PendienteAnt": num(pendientes.get("antitetica", np.nan), 2),
        "PendienteSobol": num(pendientes.get("sobol", np.nan), 2),
        "RMSEMC": num(rmse.get("mc", np.nan), 2),
        "RMSESobol": num(rmse.get("sobol", np.nan), 2),
        "NConvMax": f"{n_max:,}".replace(",", "."),
        "GananciaSobol": num((rmse.get("mc") / rmse.get("sobol")) ** 2, 1),
        "OpcionAhora": num(op["valor_ahora"]),
        "OpcionFlexible": num(op["valor_flexible"], 2),
        "OpcionFlexibleEE": num(op["valor_flexible_ee"], 2),
        "OpcionEstatico": num(op["valor_estatico"]),
        "OpcionAnioEstatico": str(op["mejor_anio_estatico"]),
        "OpcionNunca": pct(op["prob_nunca"], 1),
        "OpcionEjerce": pct(1 - op["prob_nunca"], 1),
        "MSAnalitico": num(op["validacion"]["analitico"], 4),
        "MSLSM": num(op["validacion"]["lsm"], 4),
        "MSLSMEE": num(op["validacion"]["lsm_ee"], 4),
        "MSUmbral": num(op["validacion"]["umbral"], 3),
        "MSBeta": num(op["validacion"]["beta"], 3),
        "MSError": pct(op["validacion"]["lsm"] / op["validacion"]["analitico"] - 1, 1),
        "BNCentralSens": num(sens["bn_central"]),
        "NSens": str(sens["n"]),
        "LtresDescTres": num(l3_desc.get(0.035, np.nan)),
        "LtresDescSeis": num(l3_desc.get(0.064, np.nan)),
        "CVaRSaltosBase": num(saltos.loc[BASE, "cvar_saltos"]),
        "CVaRMBGBase": num(saltos.loc[BASE, "cvar_mbg"]),
        "CVaRSaltosOpt": num(saltos.loc[opt, "cvar_saltos"]),
        "CVaRMBGOpt": num(saltos.loc[opt, "cvar_mbg"]),
        "WCambio": num(w_cambio["w"].iloc[0], 1) if not w_cambio.empty else "---",
        "WSaltos": num(umbral_saltos(saltos), 2),
        "DisenoWIntermedio": tex_nombre(ruta["diseno"].iloc[-2]),
        "DisenoWAlto": tex_nombre(ruta["diseno"].iloc[-1]),
    }
    m.update(macros_fase2013())
    lineas = ["% Archivo generado por py/generar_macros_tex.py. No editar a mano."]
    for k, v in m.items():
        lineas.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (rutas.TEX / "datos_numericos.tex").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"  {len(m)} macros escritas en tex/datos_numericos.tex")


if __name__ == "__main__":
    main()
