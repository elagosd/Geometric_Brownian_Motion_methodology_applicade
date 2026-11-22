"""Figura 1: zonas, corredores y proyectos de la red modelada."""

import numpy as np

import rutas
from estilo_figuras import COLOR_PROYECTO, NEUTRO, TEXTO, TEXTO_2, guardar, plt
from modelo.parametros import Parametros
from modelo.red import parsear_aristas


def main():
    p = Parametros()
    z = p.zonas.set_index("id")
    c = p.corredores.set_index("id")
    fig, ax = plt.subplots(figsize=(7.0, 7.4))
    for eid, e in c.iterrows():
        o, d = z.loc[e["origen"]], z.loc[e["destino"]]
        estilo = "-" if e["servicio_tp_base"] == "troncal" else "--"
        ax.plot([o.lon, d.lon], [o.lat, d.lat], estilo, color=NEUTRO, lw=1.0, zorder=1)
    desp = {"L1": -0.004, "L2": 0.004, "L3": 0.0, "RTO": 0.003, "RTN": 0.003, "L1X": -0.003, "T68": -0.002}
    for _, pr in p.proyectos.iterrows():
        col = COLOR_PROYECTO.get(pr["id"], TEXTO)
        primera = True
        for eid, phi in parsear_aristas(pr["aristas"]).items():
            e = c.loc[eid]
            o, d = z.loc[e["origen"]], z.loc[e["destino"]]
            x = np.array([o.lon, o.lon + phi * (d.lon - o.lon)]) + desp[pr["id"]]
            y = np.array([o.lat, o.lat + phi * (d.lat - o.lat)]) + desp[pr["id"]]
            estado = "comprometido" if pr["estado"] == "comprometido" else "en decisión"
            ax.plot(x, y, color=col, lw=2.6, solid_capstyle="round", zorder=2,
                    label=f"{pr['id']}: {pr['nombre']} ({estado})" if primera else None)
            primera = False
    tam = z["poblacion_miles"] / 4
    ax.scatter(z["lon"], z["lat"], s=tam, color="white", edgecolor=TEXTO, lw=1.0, zorder=3)
    for i, fila in z.iterrows():
        ax.annotate(f"{fila['codigo']}\n{fila['poblacion_miles']:.0f} mil", (fila["lon"], fila["lat"]),
                    xytext=(7, -2), textcoords="offset points", fontsize=7, color=TEXTO_2, zorder=4)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_aspect(1 / np.cos(np.radians(4.65)))
    ax.set_title("Red estilizada: 12 zonas, 18 corredores y proyectos")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.09), ncol=1, fontsize=7)
    ax.text(0.99, 0.01, "Línea continua: troncal en 2025; discontinua: bus zonal.\n"
            "Área del círculo proporcional a la población.", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color=TEXTO_2)
    guardar(fig, rutas.FIGURAS / "fig_red.pdf")


if __name__ == "__main__":
    main()
