"""Red de corredores: geometría, oferta por año y caminos mínimos.

La red es un grafo de 12 zonas y 18 corredores no dirigidos. Cada corredor
tiene una capacidad vial relativa y una oferta de transporte público (TP)
formada por servicios: el servicio base (troncal de TransMilenio con
velocidad fija o bus zonal que circula con el tráfico) y los proyectos que
lo cubren total o parcialmente. Con varios servicios, el tiempo por
kilómetro es la media armónica ponderada por capacidad:

    1 / v_e = sum_s (C_s phi_s / v_s) / sum_s C_s phi_s.

Los caminos se calculan sobre el grafo dirigido (cada corredor en ambos
sentidos) con Floyd-Warshall; la incidencia ``A[d, i*Z + j]`` vale 1 si el
arco dirigido ``d`` pertenece al camino mínimo de ``i`` a ``j``.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

RADIO_TIERRA_KM = 6371.0


def haversine(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * RADIO_TIERRA_KM * np.arcsin(np.sqrt(a)))


def longitudes(zonas: pd.DataFrame, corredores: pd.DataFrame, circuidad: float) -> np.ndarray:
    lat = zonas.set_index("id")["lat"]
    lon = zonas.set_index("id")["lon"]
    return np.array([
        circuidad * haversine(lat[o], lon[o], lat[d], lon[d])
        for o, d in zip(corredores["origen"], corredores["destino"])
    ])


def parsear_aristas(texto: str) -> dict[str, float]:
    """``"e08:1;e07:0.5"`` -> ``{"e08": 1.0, "e07": 0.5}``."""
    salida = {}
    for item in str(texto).split(";"):
        item = item.strip()
        if not item:
            continue
        eid, phi = item.split(":")
        salida[eid.strip()] = float(phi)
    return salida


@dataclass
class OfertaAnual:
    """Oferta de la red para un conjunto de proyectos abiertos.

    Todos los arreglos por arco están en el grafo dirigido (``D = 2E``).
    """

    Z: int
    origen: np.ndarray  # (D,)
    destino: np.ndarray
    longitud: np.ndarray  # km
    cap_via: np.ndarray  # veh/h por sentido
    cap_tp: np.ndarray  # pasajeros/h por sentido (total)
    tp_fijo: np.ndarray  # sum C_s phi_s / v_s de servicios con velocidad fija
    cap_zonal: np.ndarray  # capacidad del bus zonal (velocidad con el tráfico)
    frac_bus: np.ndarray  # fracción de la capacidad de TP operada con buses
    A_via: np.ndarray  # (D, Z*Z)
    A_tp: np.ndarray  # (D, Z*Z)
    n_arcos_tp: np.ndarray  # (Z*Z,)
    dist_via: np.ndarray  # (Z*Z,) km por el camino vial
    t_ref_via: np.ndarray  # (Z*Z,) min a velocidad de referencia
    t_ref_tp: np.ndarray  # (Z*Z,) min en vehículo a velocidades de referencia
    abiertos: tuple[str, ...]


def _floyd_warshall(Z: int, origen: np.ndarray, destino: np.ndarray, peso: np.ndarray):
    """Distancias y matriz de sucesores para reconstruir caminos."""
    dist = np.full((Z, Z), np.inf)
    sig = -np.ones((Z, Z), dtype=int)
    arco = -np.ones((Z, Z), dtype=int)
    np.fill_diagonal(dist, 0.0)
    for d, (o, t, w) in enumerate(zip(origen, destino, peso)):
        if w < dist[o, t]:
            dist[o, t] = w
            sig[o, t] = t
            arco[o, t] = d
    for i in range(Z):
        sig[i, i] = i
    for k in range(Z):
        mejora = dist[:, [k]] + dist[[k], :] < dist - 1e-12
        if mejora.any():
            dist = np.where(mejora, dist[:, [k]] + dist[[k], :], dist)
            sig = np.where(mejora, sig[:, [k]], sig)
    return dist, sig, arco


def _incidencia(Z: int, D: int, sig: np.ndarray, arco: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.zeros((D, Z * Z))
    n_arcos = np.zeros(Z * Z)
    for i in range(Z):
        for j in range(Z):
            if i == j:
                continue
            u = i
            pasos = 0
            while u != j:
                v = sig[u, j]
                if v < 0:
                    raise RuntimeError(f"Red desconectada entre {i} y {j}")
                A[arco[u, v], i * Z + j] = 1.0
                u = v
                pasos += 1
                if pasos > Z:
                    raise RuntimeError("Ciclo en la reconstrucción de caminos")
            n_arcos[i * Z + j] = pasos
    return A, n_arcos


_CACHE: dict = {}


def huella(p) -> tuple:
    """Clave que identifica los datos y parámetros que determinan la oferta."""
    r = p.red
    marcos = tuple(int(pd.util.hash_pandas_object(df, index=True).sum())
                   for df in (p.zonas, p.corredores, p.proyectos))
    return (r.escala_cap_via, r.circuidad, r.vel_ref_zonal, r.vel_ref_via, r.transbordo) + marcos


def construir_oferta(p, abiertos: tuple[str, ...], clave_p: tuple | None = None) -> OfertaAnual:
    """Oferta dirigida de la red con los proyectos ``abiertos`` (con memoria)."""
    abiertos = tuple(sorted(abiertos))
    clave = (clave_p if clave_p is not None else huella(p), abiertos)
    if clave in _CACHE:
        return _CACHE[clave]
    oferta = _construir(p, abiertos)
    if len(_CACHE) > 512:
        _CACHE.clear()
    _CACHE[clave] = oferta
    return oferta


def limpiar_cache():
    _CACHE.clear()


def _construir(p, abiertos: tuple[str, ...]) -> OfertaAnual:
    z, c, pr, r = p.zonas, p.corredores, p.proyectos, p.red
    Z = len(z)
    E = len(c)
    largo = longitudes(z, c, r.circuidad)
    ids = c["id"].tolist()
    pos = {eid: k for k, eid in enumerate(ids)}

    cap_tot = np.zeros(E)
    tp_fijo = np.zeros(E)
    cap_zonal = np.zeros(E)
    cap_bus = np.zeros(E)
    for k, fila in c.iterrows():
        if fila["servicio_tp_base"] == "troncal":
            cap_tot[k] += fila["cap_tp_base"]
            tp_fijo[k] += fila["cap_tp_base"] / fila["vel_tp_base"]
            cap_bus[k] += fila["cap_tp_base"]
        else:
            cap_tot[k] += fila["cap_tp_base"]
            cap_zonal[k] += fila["cap_tp_base"]
            cap_bus[k] += fila["cap_tp_base"]

    for _, proy in pr[pr["id"].isin(abiertos)].iterrows():
        for eid, phi in parsear_aristas(proy["aristas"]).items():
            k = pos[eid]
            cap = proy["capacidad"] * phi
            cap_tot[k] += cap
            tp_fijo[k] += cap / proy["velocidad"]
            if proy["clase_sobrecosto"] == "brt":
                cap_bus[k] += cap

    origen = np.concatenate([c["origen"].to_numpy(), c["destino"].to_numpy()])
    destino = np.concatenate([c["destino"].to_numpy(), c["origen"].to_numpy()])
    dup = lambda a: np.concatenate([a, a])  # noqa: E731
    largo_d = dup(largo)
    cap_via = dup(c["cap_via_rel"].to_numpy(float)) * r.escala_cap_via
    cap_tp = dup(cap_tot)
    fijo = dup(tp_fijo)
    zonal = dup(cap_zonal)
    frac_bus = dup(cap_bus / cap_tot)
    D = 2 * E

    # Caminos viales: por longitud (la red vial no cambia entre diseños).
    _, sig_v, arco_v = _floyd_warshall(Z, origen, destino, largo_d)
    A_via, _ = _incidencia(Z, D, sig_v, arco_v)
    # Caminos de TP: por tiempo en vehículo a velocidades de referencia.
    h_km = (fijo + zonal / r.vel_ref_zonal) / cap_tp
    peso_tp = largo_d * h_km * 60.0 + r.transbordo * 0.5
    _, sig_t, arco_t = _floyd_warshall(Z, origen, destino, peso_tp)
    A_tp, n_arcos = _incidencia(Z, D, sig_t, arco_t)

    dist_via = largo_d @ A_via
    t_ref_via = dist_via / r.vel_ref_via * 60.0
    t_ref_tp = (largo_d * h_km * 60.0) @ A_tp
    return OfertaAnual(Z, origen, destino, largo_d, cap_via, cap_tp, fijo, zonal, frac_bus,
                       A_via, A_tp, n_arcos, dist_via, t_ref_via, t_ref_tp, abiertos)


def abiertos_en(p, anio: int, decision: tuple[str, ...], aperturas: dict[str, int] | None = None) -> tuple[str, ...]:
    """Proyectos en operación en ``anio``: comprometidos + decisiones elegidas."""
    pr = p.proyectos.set_index("id")
    activos = []
    for pid in list(p.proyectos_comprometidos) + list(decision):
        apertura = (aperturas or {}).get(pid, int(pr.loc[pid, "apertura"]))
        if anio >= apertura:
            activos.append(pid)
    return tuple(sorted(activos))
