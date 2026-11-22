"""Estilo común de las figuras (PDF vectorial).

Paleta categórica validada para daltonismo en orden fijo; la identidad de
cada diseño o proyecto nunca depende solo del color (leyenda y etiquetas).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
REJILLA = "#e4e3df"
NEUTRO = "#9a9890"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": NEUTRO,
    "axes.labelcolor": TEXTO,
    "axes.titlecolor": TEXTO,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": REJILLA,
    "grid.linewidth": 0.6,
    "xtick.color": TEXTO_2,
    "ytick.color": TEXTO_2,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.6,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

# Color fijo por entidad (no por rango).
COLOR_PROYECTO = {"L1": CAT[0], "L2": CAT[1], "L3": CAT[2], "RTN": CAT[3], "L1X": CAT[4],
                  "RTO": CAT[5], "T68": CAT[6]}


def guardar(fig, ruta):
    fig.savefig(ruta)
    plt.close(fig)
    print(f"  figura: {ruta.name}")
