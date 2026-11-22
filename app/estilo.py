"""Hoja de estilo de la aplicación y envoltorios de gráficas y tablas."""

import inspect

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
TINTA, GRIS, LINEA = "#0b0b0b", "#52514e", "#e4e3df"
COLOR_PROYECTO = {"L1": CAT[0], "L2": CAT[1], "L3": CAT[2], "RTN": CAT[3], "L1X": CAT[4],
                  "RTO": CAT[5], "T68": CAT[6]}

pio.templates["bogota"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, sans-serif", size=13, color=TINTA),
    colorway=CAT, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor=LINEA, zeroline=False, linecolor="#9a9890"),
    yaxis=dict(gridcolor=LINEA, zeroline=False, linecolor="#9a9890"),
    hoverlabel=dict(font_size=12), margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
))
pio.templates.default = "bogota"


def _kw_ancho(funcion):
    try:
        par = inspect.signature(funcion).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return {"use_container_width": True}
    if "width" in par and isinstance(par["width"].default, str):
        return {"width": "stretch"}
    return {"use_container_width": True}


_ANCHO_GRAFICA = _kw_ancho(st.plotly_chart)
_ANCHO_TABLA = _kw_ancho(st.dataframe)


def grafica(fig, **kw):
    """Figura de Plotly a todo el ancho, con coma decimal y punto de miles."""
    fig.update_layout(separators=",.")
    fig.for_each_xaxis(lambda ax: ax.update(exponentformat="power" if ax.type == "log" else "none"))
    fig.for_each_yaxis(lambda ax: ax.update(exponentformat="power" if ax.type == "log" else "none"))
    st.plotly_chart(fig, **_ANCHO_GRAFICA, **kw)


def tabla(datos, **kw):
    kw.setdefault("hide_index", True)
    st.dataframe(datos, **_ANCHO_TABLA, **kw)


def cifra(etiqueta: str, valor: str, nota: str = "") -> str:
    return (f'<div class="cifra"><div class="et">{etiqueta}</div><div class="val">{valor}</div>'
            f'<div class="nota">{nota}</div></div>')


def rejilla(cifras: list[str]):
    st.markdown('<div class="rejilla">' + "".join(cifras) + "</div>", unsafe_allow_html=True)


def num(x, dec=1):
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
:root { --azul:#2a78d6; --tinta:#0b0b0b; --gris:#52514e; --linea:#e4e3df; --papel:#fcfcfb; }
html, body, [class*="css"] { font-family:'Inter',sans-serif; }
.stApp { background: var(--papel); }
.block-container { padding-top: 2rem; max-width: 1200px; }
h1,h2,h3 { font-family:'Crimson Pro',Georgia,serif !important; color:var(--tinta) !important; }
h1 { font-weight:700 !important; font-size:2.4rem !important; }
.portada { border-top:3px solid var(--azul); border-bottom:1px solid var(--linea);
  padding:2rem 0 1.6rem 0; margin-bottom:1.2rem; }
.portada .kicker { font-size:.75rem; letter-spacing:.16em; text-transform:uppercase; color:var(--azul);
  font-weight:600; margin-bottom:.8rem; }
.portada .sub { font-family:'Crimson Pro',serif; font-size:1.3rem; color:var(--gris); font-style:italic; }
.rejilla { display:flex; gap:.8rem; flex-wrap:wrap; margin:1rem 0; }
.cifra { flex:1 1 170px; border:1px solid var(--linea); border-radius:8px; padding:.8rem 1rem; background:white; }
.cifra .et { font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color:var(--gris); }
.cifra .val { font-family:'Crimson Pro',serif; font-size:1.9rem; font-weight:600; color:var(--tinta); line-height:1.1; }
.cifra .nota { font-size:.75rem; color:var(--gris); }
.aviso { border-left:3px solid var(--azul); background:#f3f7fd; padding:.7rem 1rem; border-radius:4px;
  font-size:.9rem; color:var(--tinta); margin:.6rem 0; }
</style>
"""


def aplicar_css():
    st.markdown(CSS, unsafe_allow_html=True)
