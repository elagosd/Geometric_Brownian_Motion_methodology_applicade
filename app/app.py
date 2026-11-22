"""Aplicación Streamlit: modelo de transporte para Bogotá con Monte Carlo y MBG.

Ejecutar desde la raíz del proyecto:   streamlit run app/app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import nucleo as N
from estilo import (CAT, COLOR_PROYECTO, GRIS, SEQ, aplicar_css, cifra, grafica, num, rejilla,
                    tabla)

st.set_page_config(page_title="Transporte Bogotá · MBG y Monte Carlo", page_icon="🚇", layout="wide")
aplicar_css()

REPO = "https://github.com/elagosd/Geometric_Brownian_Motion_methodology_applicade"


# ---------------------------------------------------------------------------
# Barra lateral: supuestos globales
# ---------------------------------------------------------------------------
def barra_lateral() -> tuple:
    if "cfg" not in st.session_state:
        st.session_state.cfg = dict(N.CONFIG_INICIAL)
    c = st.session_state.cfg
    sb = st.sidebar
    sb.markdown("### Simulación")
    c["n"] = sb.select_slider("Escenarios Monte Carlo", [64, 128, 256, 512, 1024], value=c["n"])
    c["metodo"] = sb.selectbox("Muestreo", ["sobol", "antitetica", "mc"], index=["sobol", "antitetica", "mc"].index(c["metodo"]),
                               format_func={"sobol": "Sobol (cuasi-MC)", "antitetica": "Antitéticas", "mc": "MC estándar"}.get)
    c["semilla"] = int(sb.number_input("Semilla", value=int(c["semilla"]), step=1))
    c["paso"] = sb.select_slider("Evaluar la red cada (años)", [1, 2, 3, 5], value=c["paso"],
                                 help="Con más de 1 año se interpola entre años evaluados: más rápido, menos exacto.")
    sb.markdown("### Demanda (MBG)")
    c["mu_comun"] = sb.slider("Tendencia común μc (%/año)", -1.0, 2.0, c["mu_comun"] * 100, 0.1) / 100
    c["sigma_comun"] = sb.slider("Volatilidad común σc (%/año)", 0.0, 15.0, c["sigma_comun"] * 100, 0.5) / 100
    c["saltos"] = sb.toggle("Saltos sistémicos (tipo pandemia)", value=c["saltos"])
    if c["saltos"]:
        c["lam_saltos"] = sb.slider("Intensidad (saltos/año)", 0.0, 0.2, c["lam_saltos"], 0.01)
        c["media_salto"] = sb.slider("Tamaño medio del salto (log)", -0.8, 0.0, c["media_salto"], 0.05)
    sb.markdown("### Economía")
    c["vtts"] = float(sb.slider("Valor del tiempo (COP/h)", 3000, 20000, int(c["vtts"]), 500))
    c["tasa"] = sb.slider("Tasa social de descuento (%)", 2.0, 14.0, c["tasa"] * 100, 0.1) / 100
    c["sobrecostos"] = sb.toggle("Sobrecostos de CAPEX (Flyvbjerg)", value=c["sobrecostos"])
    sb.markdown("### Política de suelo")
    c["costo_suelo"] = sb.slider("Costo anual con θ = 1 (billones)", 0.0, 2.0, c["costo_suelo"], 0.1)
    c["reduccion_nm"] = sb.slider("Viajes que pasan a pie o bici (%)", 0.0, 15.0, c["reduccion_nm"] * 100, 0.5) / 100
    c["relocalizacion"] = sb.slider("Empleo central relocalizado (%)", 0.0, 40.0, c["relocalizacion"] * 100, 1.0) / 100
    if sb.button("Restablecer supuestos"):
        st.session_state.cfg = dict(N.CONFIG_INICIAL)
        st.rerun()
    sb.caption(f"[Repositorio en GitHub]({REPO})")
    return N.clave(c)


CFG = barra_lateral()


# ---------------------------------------------------------------------------
# Páginas
# ---------------------------------------------------------------------------
def pagina_inicio():
    st.markdown("""<div class="portada"><div class="kicker">Investigación · Bogotá D.C.</div>
<h1>Modelo de transporte para Bogotá</h1>
<div class="sub">Simulación del diseño urbano óptimo con Monte Carlo y movimiento browniano geométrico</div>
<div style="margin-top:.8rem">Eduardo Lagos Durán</div></div>""", unsafe_allow_html=True)
    rejilla([
        cifra("Viajes diarios", "14,6 M", "Bogotá y 20 municipios · EM 2023"),
        cifra("Transporte público", "35 %", "de los viajes · EM 2023"),
        cifra("Velocidad en hora pico", "18,9 km/h", "TomTom Traffic Index 2025"),
        cifra("Duración media", "48 min", "por viaje · EM 2023"),
        cifra("Metro Línea 1", "marzo 2028", "72.000 pas/h-sentido · EMB"),
    ])
    st.markdown("""
La aplicación acompaña el documento de la investigación. La demanda de viajes de cada zona sigue un
**movimiento browniano geométrico multivariado** calibrado con datos de Bogotá; para cada escenario se
resuelve el equilibrio entre transporte público y vehículo privado en una red estilizada de 12 zonas y
18 corredores, y se calcula el **costo social** de cada **diseño urbano**: una combinación de líneas
férreas en decisión (extensión de la L1, L2, L3, Regiotram del Norte) y de una política de
**proximidad y policentrismo** (POT, Decreto 555 de 2021). Con miles de escenarios se estiman el
costo esperado, el CVaR y la probabilidad de beneficio neto positivo, y se elige el diseño óptimo
para distintos grados de aversión al riesgo. Una página aparte valora la **opción real** de esperar
para construir una línea de metro.
""")
    p = N.parametros(CFG)
    z, c, pr = p.zonas.set_index("id"), p.corredores.set_index("id"), p.proyectos
    fig = go.Figure()
    for _, e in c.iterrows():
        o, d = z.loc[e["origen"]], z.loc[e["destino"]]
        fig.add_trace(go.Scatter(x=[o.lon, d.lon], y=[o.lat, d.lat], mode="lines", showlegend=False,
                                 line=dict(color="#9a9890", width=1.2, dash="solid" if e["servicio_tp_base"] == "troncal" else "dot"),
                                 hoverinfo="text", text=e["nombre"]))
    from modelo.red import parsear_aristas
    for _, f in pr.iterrows():
        primera = True
        for eid, phi in parsear_aristas(f["aristas"]).items():
            e = c.loc[eid]
            o, d = z.loc[e["origen"]], z.loc[e["destino"]]
            fig.add_trace(go.Scatter(x=[o.lon, o.lon + phi * (d.lon - o.lon)], y=[o.lat, o.lat + phi * (d.lat - o.lat)],
                                     mode="lines", line=dict(color=COLOR_PROYECTO[f["id"]], width=5),
                                     name=f"{f['id']} · {f['nombre']}", legendgroup=f["id"], showlegend=primera,
                                     hovertext=f"{f['nombre']} ({f['estado']}, {f['apertura']})", hoverinfo="text"))
            primera = False
    fig.add_trace(go.Scatter(x=z["lon"], y=z["lat"], mode="markers+text", text=z["codigo"], textposition="top right",
                             marker=dict(size=np.sqrt(z["poblacion_miles"]) * 0.9, color="white",
                                         line=dict(color="#0b0b0b", width=1.2)),
                             hovertext=[f"{r.nombre}<br>{num(r.poblacion_miles, 0)} mil hab." for r in z.itertuples()],
                             hoverinfo="text", showlegend=False))
    fig.update_layout(height=620, title="Red estilizada de corredores y proyectos",
                      yaxis=dict(scaleanchor="x", scaleratio=1 / np.cos(np.radians(4.65)), title="Latitud"),
                      xaxis=dict(title="Longitud"), legend=dict(orientation="v", y=0.02, x=0.0, yanchor="bottom"))
    grafica(fig)
    tabla(pr[["id", "nombre", "estado", "apertura", "capex_billones", "capacidad", "velocidad", "fuente"]].rename(columns={
        "id": "Id", "nombre": "Proyecto", "estado": "Estado", "apertura": "Apertura", "capex_billones": "CAPEX (billones)",
        "capacidad": "Capacidad (pas/h-sentido)", "velocidad": "Velocidad (km/h)", "fuente": "Fuente / supuesto"}))


def pagina_demanda():
    st.title("Demanda estocástica: movimiento browniano geométrico")
    st.latex(r"\frac{dX_i}{X_i}=\mu_i\,dt+\sigma_c\,dW_0+s_i\,dW_i,\qquad "
             r"\ln X(t+\Delta)=\ln X(t)+\left(\mu-\tfrac12\operatorname{diag}\Sigma\right)\Delta+L\,Z\sqrt{\Delta}")
    serie = N.cargar_serie_transmilenio()
    anios, vals = serie["anio"].to_numpy(), serie["pasajeros_dia_millones"].to_numpy()
    completo = N.gbm.calibrar(anios, vals)
    excl = (anios[:-1] == 2019) & (anios[1:] == 2020)
    robusto = N.gbm.calibrar_por_tramos(anios, vals, excl)
    rejilla([cifra("σ con todos los datos", f"{num(100 * completo.sigma)} %", "dominada por el salto de 2020"),
             cifra("σ sin el salto de 2020", f"{num(100 * robusto.sigma)} %", f"μ = {num(100 * robusto.mu)} %/año"),
             cifra("Salto 2019→2020", f"{num(100 * (vals[3] / vals[2] - 1), 0)} %", "pandemia")])
    st.markdown('<div class="aviso">La calibración con pocos datos y un salto grande sobreestima la difusión; '
                'por eso el modelo separa la difusión (σ ≈ 6–7 %) de los saltos sistémicos, que se activan en la barra lateral.</div>',
                unsafe_allow_html=True)
    esc = N.escenarios(CFG)
    p = N.parametros(CFG)
    tot = esc.total(p.participacion_generacion)
    a = esc.anios
    fig = go.Figure()
    for (lo, hi), col in zip([(5, 95), (25, 75)], (SEQ[1], SEQ[3])):
        fig.add_trace(go.Scatter(x=np.r_[a, a[::-1]], y=np.r_[np.percentile(tot, hi, 0), np.percentile(tot, lo, 0)[::-1]],
                                 fill="toself", fillcolor=col, line=dict(width=0), name=f"Percentiles {lo}–{hi}", opacity=0.6))
    for k in range(min(12, tot.shape[0])):
        fig.add_trace(go.Scatter(x=a, y=tot[k], mode="lines", line=dict(color=GRIS, width=0.6), showlegend=k == 0,
                                 name="Trayectorias"))
    fig.add_trace(go.Scatter(x=a, y=np.median(tot, 0), line=dict(color=SEQ[6], width=2.5), name="Mediana"))
    fig.update_layout(title=f"Índice de demanda motorizada agregado ({esc.n} escenarios)", height=420,
                      yaxis_title="2025 = 1", hovermode="x unified")
    grafica(fig)
    Xf = esc.X[:, -1, :]
    q = np.percentile(Xf, [5, 25, 50, 75, 95], axis=0)
    fig = go.Figure()
    for i, cod in enumerate(p.zonas["codigo"]):
        fig.add_trace(go.Box(x=[cod] * 5, q1=[q[1, i]], median=[q[2, i]], q3=[q[3, i]], lowerfence=[q[0, i]],
                             upperfence=[q[4, i]], marker_color=CAT[0], showlegend=False, name=cod))
    fig.update_layout(title="Índice por zona en 2050 (caja 25–75 %, bigotes 5–95 %)", height=380, yaxis_title="2025 = 1")
    grafica(fig)
    t = a[-1] - a[0]
    teo = N.gbm.media_teorica(esc.mu, t)
    tabla(pd.DataFrame({"Zona": p.zonas["nombre"], "μ total (%/año)": np.round(100 * esc.mu, 2),
                        "E[X(2050)] teórica": np.round(teo, 4), "E[X(2050)] simulada": np.round(Xf.mean(0), 4),
                        "Error relativo (%)": np.round(100 * (Xf.mean(0) / teo - 1), 2)}))


def _diseno_ui(prefijo: str, por_defecto=("RTN",), theta=1.0):
    p = N.parametros(CFG)
    col1, col2 = st.columns([3, 1])
    proy = col1.multiselect("Proyectos en decisión", p.proyectos_decision, default=list(por_defecto), key=prefijo + "p",
                            format_func=lambda x: f"{x} · {p.proyectos.set_index('id').loc[x, 'nombre']}")
    th = col2.select_slider("Política de suelo θ", [0.0, 0.25, 0.5, 0.75, 1.0], value=theta, key=prefijo + "t")
    return tuple(sorted(proy)), th


def pagina_diseno():
    st.title("Evaluar un diseño urbano")
    st.caption("Los proyectos comprometidos (L1 en 2028, troncal Av. 68 en 2027, Regiotram de Occidente en 2030) están en todos los diseños.")
    proy, th = _diseno_ui("d")
    with st.spinner("Simulando la red en todos los escenarios…"):
        rb = N.evaluar_diseno(CFG, (), 0.0)
        rd = N.evaluar_diseno(CFG, proy, th)
    bn = rb.vp - rd.vp
    from modelo.riesgo import cvar
    rejilla([cifra("Beneficio neto esperado", f"{num(bn.mean())}", "billones COP, VP 2026, vs. base"),
             cifra("P(beneficio > 0)", f"{num(100 * np.mean(bn > 0), 0)} %", ""),
             cifra("CVaR 95 % del costo", num(cvar(rd.vp)), f"base: {num(cvar(rb.vp))}"),
             cifra("Tiempo medio 2050", f"{num(rd.tiempo_medio[:, -1].mean())} min", f"base: {num(rb.tiempo_medio[:, -1].mean())}"),
             cifra("CAPEX esperado", num(rd.capex_total.mean()), "billones con sobrecostos")])
    fig = go.Figure(go.Histogram(x=bn, nbinsx=50, marker_color=CAT[0], marker_line=dict(color="white", width=0.5)))
    fig.add_vline(0, line_color="#0b0b0b")
    fig.add_vline(bn.mean(), line_dash="dash", line_color=GRIS, annotation_text="media")
    fig.update_layout(title="Distribución del beneficio neto por escenario", xaxis_title="billones COP", height=340,
                      showlegend=False)
    grafica(fig)
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Tiempo medio de viaje (min)", "Velocidad vial (km/h)",
                                                        "Participación del TP", "Hacinamiento máximo del TP"))
    for (r, cidx, attr) in ((1, 1, "tiempo_medio"), (1, 2, "vel_via"), (2, 1, "part_tp"), (2, 2, "hac_max")):
        for res, nom, col in ((rb, "Base", GRIS), (rd, "Diseño", CAT[0])):
            y = getattr(res, attr)
            fig.add_trace(go.Scatter(x=np.r_[res.anios, res.anios[::-1]], y=np.r_[np.percentile(y, 95, 0), np.percentile(y, 5, 0)[::-1]],
                                     fill="toself", fillcolor=col, opacity=0.15, line=dict(width=0), showlegend=False), r, cidx)
            fig.add_trace(go.Scatter(x=res.anios, y=np.median(y, 0), line=dict(color=col, width=2), name=nom,
                                     showlegend=(r, cidx) == (1, 1)), r, cidx)
    fig.update_layout(height=620, title="Mediana y banda 5–95 %")
    grafica(fig)
    etiquetas = {"tiempo": "Tiempo", "auto": "Auto", "bus": "Buses", "om": "O&M férrea", "suelo": "Suelo",
                 "capex": "CAPEX", "residual": "Residual"}
    dif = pd.DataFrame({"Componente": list(etiquetas.values()),
                        "Base": [rb.vp_componente(k).mean() for k in etiquetas],
                        "Diseño": [rd.vp_componente(k).mean() for k in etiquetas]})
    dif["Diferencia"] = dif["Diseño"] - dif["Base"]
    tabla(dif.round(2))


def pagina_optimizacion():
    st.title("Diseño urbano óptimo bajo riesgo")
    st.latex(r"\min_{d\in\mathcal D}\;(1-w)\,\mathbb E_n[VP_d]+w\,\mathrm{CVaR}_{95\%,n}[VP_d]"
             r"\quad\text{s.a.}\quad \mathbb E_n[\mathrm{CAPEX}_d]\le B")
    st.caption("Se evalúan los 48 diseños (16 combinaciones de proyectos × 3 niveles de θ) con los mismos escenarios.")
    if st.button("Evaluar los 48 diseños", type="primary") or st.session_state.get("opt_listo") == CFG:
        barra = st.progress(0.0)
        N.evaluar_todos(CFG, barra)
        barra.empty()
        st.session_state.opt_listo = CFG
        df, vp = N.tabla_resumen(CFG)
        c1, c2 = st.columns(2)
        w = c1.slider("Aversión al riesgo w", 0.0, 1.0, 0.0, 0.05)
        presupuesto = c2.slider("Presupuesto de CAPEX esperado (billones)", 0.0, float(df["capex_medio"].max()) + 1,
                                float(df["capex_medio"].max()) + 1, 1.0)
        from modelo.optimizacion import elegir
        e = elegir(df, w, presupuesto)
        rejilla([cifra("Diseño elegido", e["diseno"], f"w = {w:g}"),
                 cifra("E[VP]", num(e["vp_media"]), "billones COP"),
                 cifra("CVaR 95 %", num(e["vp_cvar"]), "billones COP"),
                 cifra("Beneficio neto", num(e["bn_media"]), f"P(>0) = {num(100 * e['prob_bn_pos'], 0)} %")])
        fig = go.Figure()
        for th, sim in ((0.0, "circle"), (0.5, "square"), (1.0, "diamond")):
            g = df[df["theta"] == th]
            fig.add_trace(go.Scatter(x=g["vp_media"], y=g["vp_cvar"], mode="markers", name=f"θ = {th:g}",
                                     marker=dict(symbol=sim, size=10, color="white", line=dict(color=GRIS, width=1.2)),
                                     text=g["diseno"], hovertemplate="%{text}<br>E=%{x:.1f}<br>CVaR=%{y:.1f}<extra></extra>"))
        par = df[df["pareto"]].sort_values("vp_media")
        fig.add_trace(go.Scatter(x=par["vp_media"], y=par["vp_cvar"], mode="lines+markers", name="Frontera de Pareto",
                                 line=dict(color=CAT[0], width=2), marker=dict(size=8), text=par["diseno"],
                                 hovertemplate="%{text}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[e["vp_media"]], y=[e["vp_cvar"]], mode="markers", name="Elegido",
                                 marker=dict(size=16, color=CAT[1], symbol="star")))
        fig.update_layout(title="Costo esperado frente a CVaR (menor es mejor)", height=480,
                          xaxis_title="E[VP] (billones COP)", yaxis_title="CVaR 95 % (billones COP)")
        grafica(fig)
        cols = {"diseno": "Diseño", "vp_media": "E[VP]", "vp_cvar": "CVaR95", "bn_media": "E[BN]",
                "prob_bn_pos": "P(BN>0)", "capex_medio": "CAPEX", "tiempo_2050": "Tiempo 2050", "pareto": "Pareto"}
        tabla(df[list(cols)].rename(columns=cols).round(3), height=420)
        top = df.head(5)["diseno"].tolist()
        fig = go.Figure()
        for nom, col in zip(top, CAT):
            fig.add_trace(go.Violin(y=vp[BASE_NOM] - vp[nom], name=nom, line_color=col, meanline_visible=True, points=False))
        fig.add_hline(0, line_color="#0b0b0b")
        fig.update_layout(title="Beneficio neto de los cinco diseños de menor costo esperado", height=420,
                          yaxis_title="billones COP", showlegend=False)
        grafica(fig)
    else:
        st.info("Pulse el botón para evaluar todos los diseños con los supuestos de la barra lateral. "
                "Con 256 escenarios y evaluación cada 2 años tarda alrededor de un minuto.")


BASE_NOM = N.BASE


def pagina_opciones():
    st.title("Opciones reales: ¿cuándo construir una línea de metro?")
    st.markdown("Decidir en el año τ implica construir durante los años de obra y abrir después. Con mínimos cuadrados "
                "de Monte Carlo (Longstaff–Schwartz) se estima, en cada año, si conviene decidir ya o esperar, usando como "
                "variable de estado el ahorro operativo anual que la línea produciría si estuviera abierta.")
    p = N.parametros(CFG)
    c1, c2, c3 = st.columns(3)
    proyecto = c1.selectbox("Proyecto", ["L3", "L2", "L1X", "RTN"])
    otros = c2.multiselect("Resto del diseño", [x for x in p.proyectos_decision if x != proyecto], default=["RTN"] if proyecto != "RTN" else [])
    theta = c3.select_slider("θ", [0.0, 0.5, 1.0], value=1.0)
    primer, ultimo = st.slider("Ventana de decisión", 2027, 2045, (2027, 2040))
    with st.spinner("Calculando beneficios potenciales y la regla de ejercicio…"):
        anios, B, res = N.opcion_real(CFG, proyecto, tuple(sorted(otros)), theta, primer, ultimo)
    rejilla([cifra("Comprometerse en " + str(primer), num(res.valor_ahora), "VPN esperado, billones"),
             cifra("Mejor año fijo", str(res.mejor_anio_estatico), num(res.valor_estatico)),
             cifra("Regla flexible (LSM)", num(res.valor_flexible, 2), f"± {num(res.valor_flexible_ee, 2)}"),
             cifra("Escenarios donde nunca se construye", f"{num(100 * res.prob_nunca, 1)} %", "dentro de la ventana")])
    fig = make_subplots(rows=1, cols=2, subplot_titles=("VPN esperado según el año de decisión", "Ahorro operativo anual potencial"))
    fig.add_trace(go.Scatter(x=res.anios_decision, y=res.vpn_por_anio, mode="lines+markers", name="Año fijo",
                             line=dict(color=CAT[0])), 1, 1)
    fig.add_hline(res.valor_flexible, line_color=CAT[1], row=1, col=1, annotation_text="flexible")
    fig.add_trace(go.Scatter(x=np.r_[anios, anios[::-1]], y=np.r_[np.percentile(B, 95, 0), np.percentile(B, 5, 0)[::-1]],
                             fill="toself", fillcolor=SEQ[1], line=dict(width=0), name="5–95 %"), 1, 2)
    fig.add_trace(go.Scatter(x=anios, y=np.median(B, 0), line=dict(color=SEQ[6]), name="Mediana"), 1, 2)
    ok = ~np.isnan(res.frontera)
    fig.add_trace(go.Scatter(x=res.anios_decision[ok], y=res.frontera[ok], mode="markers", name="Frontera de ejercicio",
                             marker=dict(color=CAT[1], symbol="square", size=8)), 1, 2)
    fig.update_layout(height=420)
    grafica(fig)
    st.subheader("Validación: McDonald y Siegel (1986)")
    c = st.columns(5)
    V0 = c[0].number_input("V₀", value=1.0)
    I = c[1].number_input("I", value=1.0)
    mu = c[2].number_input("μ", value=0.02, format="%.3f")
    sig = c[3].number_input("σ", value=0.20, format="%.3f")
    r = c[4].number_input("r", value=0.06, format="%.3f")
    try:
        (F, vstar, beta), (Fl, ee) = N.validacion_ms(V0, I, mu, sig, r, 10000)
        rejilla([cifra("Analítico F(V₀)", num(F, 4), f"V* = {num(vstar, 3)}, β = {num(beta, 3)}"),
                 cifra("LSM", num(Fl, 4), f"± {num(ee, 4)}"),
                 cifra("Diferencia", f"{num(100 * (Fl / F - 1), 1)} %", "sesgo bajo esperado del LSM")])
    except ValueError as err:
        st.error(str(err))


def pagina_convergencia():
    st.title("Convergencia del Monte Carlo")
    st.markdown("Error del estimador del costo esperado del diseño base según el método de muestreo. "
                "El MC estándar converge como n^(-1/2); las antitéticas reducen la varianza y la secuencia de "
                "Sobol aleatorizada mejora la tasa cuando el integrando es suave.")
    reps = st.slider("Repeticiones por tamaño", 4, 20, 8)
    if st.button("Correr el experimento"):
        with st.spinner("Corriendo réplicas…"):
            df = N.convergencia(CFG, (32, 64, 128, 256), reps)
        fig = go.Figure()
        for (metodo, g), col in zip(df.groupby("metodo", sort=False), CAT):
            fig.add_trace(go.Scatter(x=g["n"], y=g["desv"], mode="lines+markers", name=metodo, line=dict(color=col)))
        n = np.array([32, 256])
        d0 = df[df["metodo"] == "mc"]["desv"].iloc[0]
        fig.add_trace(go.Scatter(x=n, y=d0 * (n / 32) ** -0.5, line=dict(dash="dot", color=GRIS), name="n^(-1/2)"))
        fig.update_layout(xaxis_type="log", yaxis_type="log", height=420, xaxis_title="n",
                          yaxis_title="Desviación entre réplicas de E[VP]", title="Error de Monte Carlo")
        grafica(fig)
        tabla(df.round(4))


def pagina_metodo():
    st.title("Metodología y fuentes")
    st.markdown(f"""
**Modelo.** 12 zonas (9 de Bogotá agregando localidades y 3 de la región), 18 corredores. Demanda motorizada en hora
pico de 880.000 viajes, distribuida con un modelo gravitacional doblemente acotado; partición modal logit entre TP y
vehículo privado con congestión BPR y hacinamiento; equilibrio por promedios sucesivos autorregulados.

**Calibración del año base (2025).** La escala de capacidad vial, la constante modal del TP y el parámetro gravitacional
se ajustan para reproducir 18,9 km/h en hora pico (TomTom 2025), 35/60 de participación del TP entre motorizados
(EM 2023) y 48 minutos de duración media (EM 2023).

**Costo social.** Tiempo valorado al VTTS, operación y externalidades del auto, operación de buses, O&M férrea, política
de suelo y CAPEX con sobrecostos lognormales (Flyvbjerg, 2005); descuento a la tasa social del DNP (9 %) y valor residual
en 2050.

**Riesgo.** VaR y CVaR al 95 % (Rockafellar y Uryasev, 2000); frontera de Pareto media–CVaR; opciones reales con LSM.

**Limitaciones.** Es un modelo estratégico estilizado: no reemplaza el modelo de cuatro etapas de la Secretaría de
Movilidad. Varias poblaciones zonales, capacidades y costos son aproximaciones del autor, señaladas en el documento y
editables en `datos/`.

Código, documento y datos: [{REPO}]({REPO})
""")
    st.subheader("Fuentes de los datos")
    tabla(pd.DataFrame({"Dato": list(N.FUENTES.keys()), "Fuente": list(N.FUENTES.values())}))



# ---------------------------------------------------------------------------
# Fase I: la coyuntura de 2013
# ---------------------------------------------------------------------------
ELEV, SUB_OF, SUB_UN = "Elevado | θ=0", "Subterráneo (costo oficial) | θ=0", "Subterráneo (costo unitarios) | θ=0"


def pagina_fase1():
    st.title("Fase I · Bogotá en 2013: la decisión de la Primera Línea")
    st.markdown("En 2013 Bogotá contrató el diseño de un metro **subterráneo** (27 km, 27 estaciones, 2014); en 2016–2017 "
                "se cambió a un metro **elevado** hasta la calle 72 (CONPES 3900). Esta página muestra el modelo "
                "recalibrado a 2013: escenarios de demanda con la información de entonces, evaluación *ex ante* y "
                "evaluación *ex post* con la demanda que realmente ocurrió. Los resultados provienen de "
                "`py/correr_fase_2013.py`.")
    try:
        R = N.resultados_fase1()
    except FileNotFoundError as err:
        st.error(str(err))
        return
    cal, res = R["calibracion"], R["resumen"]
    rejilla([cifra("Velocidad pico 2013", f"{num(cal['indicadores']['vel_via'])} km/h", "calibrada"),
             cifra("Transporte público", f"{num(100 * cal['indicadores']['part_tp'])} %", "de los viajes motorizados"),
             cifra("Tasa de descuento", "12 %", "vigente en 2013 (hoy 9 %)"),
             cifra("Demanda observada 2025", f"percentil {num(100 * res['percentil_observado_2025'])}", "de los escenarios de 2013")])
    if "escenarios" in R:
        E = R["escenarios"]
        a, tot, obs = E["anios"], E["total"], E["observado"]
        k = a <= 2035
        fig = go.Figure()
        for lo, hi, col, nom in ((5, 95, SEQ[1], "Escenarios 2013: 5–95 %"), (25, 75, SEQ[3], "25–75 %")):
            fig.add_trace(go.Scatter(x=np.r_[a[k], a[k][::-1]], y=np.r_[np.percentile(tot[:, k], hi, 0), np.percentile(tot[:, k], lo, 0)[::-1]],
                                     fill="toself", fillcolor=col, line=dict(width=0), name=nom))
        fig.add_trace(go.Scatter(x=a[k], y=np.median(tot[:, k], 0), line=dict(color=SEQ[6]), name="Mediana vista desde 2013"))
        fig.add_trace(go.Scatter(x=np.arange(2013, 2013 + obs.size), y=obs, mode="lines+markers", line=dict(color=CAT[1]),
                                 name="Trayectoria observada (estilizada)"))
        fig.update_layout(height=400, title="Lo que se esperaba en 2013 y lo que ocurrió", yaxis_title="Índice (2013 = 1)")
        grafica(fig)
    st.subheader("Evaluación ex ante (información de 2013)")
    tasa = st.radio("Tasa de descuento", [0.12, 0.09, 0.064], format_func=lambda t: f"{num(100 * t)} %", horizontal=True,
                    key="tasa_fase1")
    d = R["fase2013_descuento"]
    d = d[(d["tasa"] == tasa) & (d["diseno"] != "Sin metro | θ=0")].copy()
    fig = go.Figure(go.Bar(x=d["bn_media"], y=d["diseno"], orientation="h",
                           marker_color=[CAT[2] if v > 0 else CAT[7] for v in d["bn_media"]],
                           text=[num(v) for v in d["bn_media"]], textposition="outside"))
    fig.update_layout(height=380, xaxis_title="Beneficio neto esperado vs. no construir metro (billones de 2025, VP 2014)")
    grafica(fig)
    tabla(d.rename(columns={"diseno": "Diseño", "bn_media": "E[BN]", "prob_bn_pos": "P(BN>0)"})[["Diseño", "E[BN]", "P(BN>0)"]].round(3))
    st.subheader("Evaluación ex post (demanda observada, tasa 12 %)")
    ep = R["fase2013_expost"]
    tabla(ep[ep["diseno"] != "Sin metro"][["diseno", "capex_medio", "bn_media", "prob_bn_pos"]]
          .rename(columns={"diseno": "Alternativa (apertura)", "capex_medio": "CAPEX medio", "bn_media": "E[BN]",
                           "prob_bn_pos": "P(BN>0)"}).round(3))
    st.caption("Con la demanda observada ninguna primera línea se paga al 12 %; el subterráneo pierde más que el elevado "
               "con cualquier fecha de apertura.")


# ---------------------------------------------------------------------------
# Metro: elevado vs. subterráneo
# ---------------------------------------------------------------------------
def pagina_metro():
    st.title("¿Elevado o subterráneo? Mapa de decisión del metro")
    st.markdown("El modelo de transporte no ve el **costo urbano del viaducto** (ruido, sombra, efecto barrera, pérdida de "
                "valor de los predios). Esta página lo agrega como un costo anual $E$ desde la apertura y muestra cuándo "
                "conviene cada tipología. Mueva los supuestos y observe en qué punto cambia la decisión.")
    try:
        R = N.resultados_fase1()
    except FileNotFoundError as err:
        st.error(str(err))
        return
    c1, c2 = st.columns(2)
    tasa = c1.radio("Tasa de descuento", [0.12, 0.09, 0.064], format_func=lambda t: f"{num(100 * t)} %", horizontal=True,
                    key="tasa_metro")
    ext = c2.slider("Costo urbano del viaducto E (billones por año)", 0.0, 6.0, 0.20, 0.01,
                    help="Estimación por pérdida de valor de los predios: 0,06–0,28 billones por año.")
    d = R["fase2013_descuento"].set_index(["tasa", "diseno"])["bn_media"]
    u = R["fase2013_umbral"]
    u = u[u["tasa"] == tasa].set_index("alternativa")
    suma = float(u.loc[SUB_OF, "diferencia_vp"] / u.loc[SUB_OF, "externalidad_umbral"])
    bn_e = d[(tasa, ELEV)] - ext * suma
    filas = [("Elevado (con costo urbano E)", bn_e), ("Subterráneo, costo oficial 2014", d[(tasa, SUB_OF)]),
             ("Subterráneo, costos unitarios 2023", d[(tasa, SUB_UN)])]
    mejor = max(filas, key=lambda f: f[1])[0]
    rejilla([cifra(n, num(v), "E[BN] billones, VP 2014") for n, v in filas] +
            [cifra("Decisión", mejor.split(" (")[0].split(",")[0], f"con E = {num(ext, 2)}")])
    rejilla([cifra("Umbral E* · costo oficial", num(u.loc[SUB_OF, "externalidad_umbral"], 2), "billones/año"),
             cifra("Umbral E* · costos unitarios", num(u.loc[SUB_UN, "externalidad_umbral"], 2), "billones/año"),
             cifra("Rango estimado de E", "0,06 – 0,28", "pérdida de valor de los predios")])
    x = np.linspace(0, 6, 121)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=d[(tasa, ELEV)] - x * suma, name="Elevado", line=dict(color=CAT[0])))
    for nom, col in ((SUB_OF, CAT[1]), (SUB_UN, CAT[2])):
        fig.add_hline(d[(tasa, nom)], line_color=col, annotation_text=nom.replace(" | θ=0", ""))
    fig.add_vrect(0.06, 0.28, fillcolor=GRIS, opacity=0.25, line_width=0, annotation_text="rango estimado")
    fig.add_vline(ext, line_dash="dot")
    fig.update_layout(height=420, xaxis_title="Costo urbano del viaducto E (billones por año)",
                      yaxis_title="Beneficio neto esperado (billones)")
    grafica(fig)
    st.subheader("Costos unitarios por tipología (estudio EMB, 2023)")
    c1, c2 = st.columns(2)
    km = c1.number_input("Longitud (km)", 1.0, 60.0, 23.96)
    est = c2.number_input("Estaciones", 1, 60, 16)
    c = N.costos_metro(km, int(est))
    rejilla([cifra("CAPEX elevado", num(c["capex_elevado"]), "billones de 2025"),
             cifra("CAPEX subterráneo", num(c["capex_subterraneo"]), "billones de 2025"),
             cifra("Prima del subterráneo", f"{num(100 * c['prima'], 0)} %", ""),
             cifra("Km elevados por km subterráneo", num(c["capex_subterraneo"] / c["capex_elevado"], 2), "mismo presupuesto")])
    st.subheader("Líneas futuras (fase II, tasa 9 %)")
    t = R["fase2_tipologia"]
    tabla(t[["diseno", "capex_medio", "bn_media", "prob_bn_pos", "externalidad_umbral"]]
          .rename(columns={"diseno": "Diseño", "capex_medio": "CAPEX medio", "bn_media": "E[BN]", "prob_bn_pos": "P(BN>0)",
                           "externalidad_umbral": "Umbral E* (bill./año)"}).round(3))
    st.info("**Conclusión del modelo.** Con la tasa de 12 % de la época el elevado era preferible salvo que el costo urbano "
            "del viaducto superara ~0,9 billones/año (costo oficial). Con la tasa actual de 9 % y el costo oficial de 2014 "
            "ambas tipologías empatan (umbral ~0,23), y por debajo de ~8,2 % gana el subterráneo. El tamaño de Bogotá, "
            "por sí solo, no lo justifica: la capacidad no es la restricción y una red de varias líneas favorece los "
            "kilómetros más baratos.")


paginas = [
    st.Page(pagina_inicio, title="Inicio", icon="🏙️", default=True),
    st.Page(pagina_demanda, title="Demanda (MBG)", icon="📈", url_path="demanda"),
    st.Page(pagina_diseno, title="Evaluar un diseño", icon="🚇", url_path="diseno"),
    st.Page(pagina_optimizacion, title="Diseño óptimo", icon="🎯", url_path="optimo"),
    st.Page(pagina_opciones, title="Opciones reales", icon="⏳", url_path="opciones"),
    st.Page(pagina_fase1, title="Fase I · 2013", icon="🕰️", url_path="fase-2013"),
    st.Page(pagina_metro, title="Metro: elevado vs. subterráneo", icon="🚝", url_path="metro"),
    st.Page(pagina_convergencia, title="Convergencia MC", icon="🎲", url_path="convergencia"),
    st.Page(pagina_metodo, title="Metodología", icon="📚", url_path="metodologia"),
]
st.navigation(paginas).run()
