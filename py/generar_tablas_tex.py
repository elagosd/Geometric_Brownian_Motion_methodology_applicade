"""Genera las tablas del documento en tex/tablas/."""

import json

import pandas as pd

import rutas
from generar_macros_tex import num, pct, tex_nombre
from modelo.parametros import parametros_calibrados

S, T = rutas.SALIDAS, rutas.TABLAS
BASE = "Base | θ=0"


def escribir(nombre, texto):
    (T / nombre).write_text(texto, encoding="utf-8")
    print(f"  tabla: {nombre}")


def tabla_disenos():
    df = pd.read_csv(S / "resumen_disenos.csv")
    filas = []
    for k, f in df.iterrows():
        if k >= 12 and f["diseno"] not in (BASE, "L1X+L2+L3+RTN | θ=1", "L3 | θ=0", "L2 | θ=0"):
            continue
        filas.append(
            f"{tex_nombre(f['diseno'])} & {num(f['vp_media'])} & {num(f['vp_cvar'])} & "
            f"{num(f['bn_media'])} & {pct(f['prob_bn_pos'])} & {num(f['capex_medio'])} & "
            f"{num(f['tiempo_2050'])} & {'\\checkmark' if f['pareto'] else ''} \\\\")
    cuerpo = "\n".join(filas)
    escribir("tabla_disenos.tex", rf"""\begin{{tabular}}{{p{{4.1cm}}rrrrrrc}}
\toprule
Diseño & E[VP] & CVaR$_{{95}}$ & E[BN] & P(BN$>$0) & CAPEX & $\bar t_{{2050}}$ & Pareto \\
 & \multicolumn{{3}}{{c}}{{billones COP}} & & billones & min & \\
\midrule
{cuerpo}
\bottomrule
\end{{tabular}}
""")


def tabla_proyectos():
    p = parametros_calibrados()
    filas = []
    for _, f in p.proyectos.iterrows():
        estado = "comprometido" if f["estado"] == "comprometido" else "en decisión"
        capex = "—" if f["capex_billones"] == 0 else num(f["capex_billones"])
        filas.append(f"{f['id']} & {f['nombre']} & {estado} & {f['apertura']} & {capex} & "
                     f"{num(f['capacidad'], 0)} & {num(f['velocidad'], 0)} \\\\")
    escribir("tabla_proyectos.tex", r"""\begin{tabular}{llllrrr}
\toprule
Id & Proyecto & Estado & Apertura & CAPEX & Capacidad & Velocidad \\
 & & & & billones & pas/h-sentido & km/h \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}
""")


def tabla_zonas():
    p = parametros_calibrados()
    filas = []
    for _, z in p.zonas.iterrows():
        mu = z["mu_demografica"] + p.demanda.mu_comun
        filas.append(f"{z['codigo']} & {z['nombre']} & {z['ambito']} & {num(z['poblacion_miles'], 0)} & "
                     f"{pct(mu, 1)} & {pct(z['sigma_idio'], 0)} & {pct(z['atraccion'], 1)} \\\\")
    escribir("tabla_zonas.tex", r"""\begin{tabular}{lllrrrr}
\toprule
Código & Zona & Ámbito & Población (miles) & $\mu_i$ & $s_i$ & Atracción \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}
""")


def tabla_saltos():
    df = pd.read_csv(S / "saltos.csv")
    filas = [f"{tex_nombre(f['diseno'])} & {num(f['media_mbg'])} & {num(f['cvar_mbg'])} & "
             f"{num(f['media_saltos'])} & {num(f['cvar_saltos'])} \\\\" for _, f in df.iterrows()]
    escribir("tabla_saltos.tex", r"""\begin{tabular}{p{4.6cm}rrrr}
\toprule
 & \multicolumn{2}{c}{MBG} & \multicolumn{2}{c}{MBG con saltos} \\
Diseño & E[VP] & CVaR$_{95}$ & E[VP] & CVaR$_{95}$ \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}
""")


def tabla_descuento():
    df = pd.read_csv(S / "descuento.csv")
    piv = df.pivot(index="diseno", columns="tasa", values="bn_media")
    filas = [tex_nombre(d) + " & " + " & ".join(num(v) for v in fila.values) + r" \\" for d, fila in piv.iterrows()]
    cab = " & ".join(pct(t, 1) for t in piv.columns)
    escribir("tabla_descuento.tex", rf"""\begin{{tabular}}{{p{{4.6cm}}rrrr}}
\toprule
Diseño & {cab} \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}
""")


def tabla_calibracion():
    cal = json.loads((S / "calibracion_red.json").read_text(encoding="utf-8"))
    i, m = cal["indicadores"], cal["metas"]
    escribir("tabla_calibracion.tex", rf"""\begin{{tabular}}{{lrrl}}
\toprule
Indicador del año base & Meta & Modelo & Fuente de la meta \\
\midrule
Velocidad vial en hora pico (km/h) & {num(m['vel_via'])} & {num(i['vel_via'], 2)} & TomTom Traffic Index 2025 \\
Participación del TP entre motorizados & {pct(m['part_tp'], 1)} & {pct(i['part_tp'], 1)} & EM 2023 (35 \% de 60 \%) \\
Duración media del viaje (min) & {num(m['tiempo_medio'])} & {num(i['tiempo_medio'], 2)} & EM 2023 \\
\midrule
Parámetro calibrado & \multicolumn{{2}}{{r}}{{Valor}} & \\
\midrule
Escala de capacidad vial (veh/h) & \multicolumn{{2}}{{r}}{{{num(cal['escala_cap_via'], 0)}}} & \\
Constante modal del TP (min) & \multicolumn{{2}}{{r}}{{{num(cal['asc_tp'], 2)}}} & \\
Parámetro gravitacional (1/min) & \multicolumn{{2}}{{r}}{{{num(cal['beta_gravedad'], 4)}}} & \\
\bottomrule
\end{{tabular}}
""")


def tabla_fase2013():
    """Evaluación ex ante (decisión en 2014) a tres tasas de descuento."""
    de = pd.read_csv(S / "fase2013_descuento.csv")
    ex = pd.read_csv(S / "fase2013_exante.csv").set_index("diseno")
    filas = []
    for d in ex.index:
        if d.startswith("Sin metro | θ=0"):
            continue
        g = de[de["diseno"] == d].set_index("tasa")
        f = ex.loc[d]
        filas.append(f"{tex_nombre(d)} & {num(f['capex_medio'])} & {num(g.loc[0.064, 'bn_media'])} & "
                     f"{num(g.loc[0.09, 'bn_media'])} & {num(g.loc[0.12, 'bn_media'])} & {pct(g.loc[0.12, 'prob_bn_pos'])} & "
                     f"{num(f['tiempo_2030'])} \\\\")
    escribir("tabla_fase2013.tex", r"""\begin{tabular}{p{5.2cm}rrrrrr}
\toprule
Diseño (visto desde 2013) & CAPEX & \multicolumn{3}{c}{E[BN] a la tasa} & P(BN$>$0) & $\bar t_{2030}$ \\
\cmidrule(lr){3-5}
 & medio & 6,4\,\% & 9\,\% & 12\,\% & al 12\,\% & min \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}
""")


def tabla_expost():
    xp = pd.read_csv(S / "fase2013_expost.csv")
    filas = [f"{f['diseno']} & {num(f['capex_medio'])} & {num(f['bn_media'])} & {num(f['bn_ee'], 2)} & "
             f"{pct(f['prob_bn_pos'], 1)} & {num(f['tiempo_2030'])} \\\\" for _, f in xp.iterrows() if f["diseno"] != "Sin metro"]
    escribir("tabla_expost.tex", r"""\begin{tabular}{p{6.2cm}rrrrr}
\toprule
Alternativa (año de apertura) & CAPEX & E[BN] & e.e. & P(BN$>$0) & $\bar t_{2030}$ \\
 & \multicolumn{3}{c}{billones COP de 2025} & & min \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}
""")


def tabla_metro_costos():
    mc = pd.read_csv(S / "metro_costos.csv")
    filas = [f"{f['linea']} & {num(f['km'])} & {int(f['estaciones'])} & {num(f['capex_elevado'])} & "
             f"{num(f['capex_subterraneo'])} & {pct(f['prima'])} & {num(f['om_elevado'], 2)} & {num(f['om_subterraneo'], 2)} \\\\"
             for _, f in mc.iterrows()]
    escribir("tabla_metro_costos.tex", r"""\begin{tabular}{p{4.6cm}rrrrrrr}
\toprule
Línea & km & Est. & \multicolumn{2}{c}{CAPEX (billones)} & Prima & \multicolumn{2}{c}{O\&M (billones/año)} \\
\cmidrule(lr){4-5}\cmidrule(lr){7-8}
 & & & Elevado & Subterráneo & & Elevado & Subterráneo \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}
""")


def tabla_tipologia():
    ti = pd.read_csv(S / "fase2_tipologia.csv")
    filas = []
    for _, f in ti.iterrows():
        if f["diseno"] == BASE:
            continue
        u = "" if pd.isna(f["externalidad_umbral"]) else num(f["externalidad_umbral"], 2)
        filas.append(f"{tex_nombre(f['diseno'])} & {num(f['capex_medio'])} & {num(f['bn_media'])} & "
                     f"{pct(f['prob_bn_pos'])} & {num(f['vp_cvar'])} & {u} \\\\")
    escribir("tabla_tipologia.tex", r"""\begin{tabular}{p{4.2cm}rrrrr}
\toprule
Diseño & CAPEX & E[BN] & P(BN$>$0) & CVaR$_{95}$ & Umbral \\
 & \multicolumn{2}{c}{billones} & & billones & bill./año \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}
""")


if __name__ == "__main__":
    tabla_fase2013()
    tabla_expost()
    tabla_metro_costos()
    tabla_tipologia()
    tabla_disenos()
    tabla_proyectos()
    tabla_zonas()
    tabla_saltos()
    tabla_descuento()
    tabla_calibracion()
