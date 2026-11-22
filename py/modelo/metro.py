"""Costos de líneas de metro elevadas y subterráneas.

Los costos unitarios provienen del estudio de la Empresa Metro de Bogotá y
la Alcaldía presentado en enero de 2023 para evaluar la subterranización de
tramos de la Línea 1 («Comparativo de costos unitarios de metro elevado y
subterráneo», pesos constantes de noviembre de 2022). Se indexan a pesos de
2025 con la inflación de 2023 (9,28 %), 2024 (5,20 %) y 2025 (5,10 %).

    CAPEX = km * (obra civil por km + sistemas y material rodante por km)
            + estaciones * costo por estación + otros costos

Los «otros costos» (patio-taller, predios, redes, interventoría) no están en
la tabla de costos unitarios; se toman de la diferencia entre el CAPEX total
de la Línea 1 elevada y sus componentes unitarios, y se suponen iguales para
ambas tipologías.
"""

from __future__ import annotations

from dataclasses import dataclass

INDEXACION_2022_2025 = 1.0928 * 1.0520 * 1.0510  # ≈ 1,208

# Millones de pesos de noviembre de 2022 (estudio EMB, enero de 2023).
UNITARIOS_2022 = {
    "elevado": {"civil_km": 161_947, "sistemas_km": 241_043, "estacion": 129_742, "om_km_anio": 10_388},
    "subterraneo": {"civil_km": 264_609, "sistemas_km": 284_257, "estacion": 706_684, "om_km_anio": 14_318},
}

# CAPEX totales (millones de pesos de noviembre de 2022) del mismo estudio.
CAPEX_2022 = {
    "subterraneo_2015": {"capex": 24_731_796, "km": 31.64},
    "elevado_factibilidad_2017": {"capex": 17_454_730, "km": 23.96},
    "elevado_concesion_2019": {"capex": 16_817_946, "km": 23.96},
    "l2_subterraneo_2021": {"capex": 16_249_902, "km": 15.50},
}

ESTACIONES_L1 = 16


def unitarios_2025(tipo: str) -> dict[str, float]:
    """Costos unitarios en billones de pesos de 2025."""
    return {k: v * INDEXACION_2022_2025 / 1e6 for k, v in UNITARIOS_2022[tipo].items()}


def capex_total_2025(clave: str) -> float:
    return CAPEX_2022[clave]["capex"] * INDEXACION_2022_2025 / 1e6


def otros_costos_l1() -> float:
    """Costos de la L1 elevada no cubiertos por los unitarios (billones de 2025)."""
    u = unitarios_2025("elevado")
    comp = 23.96 * (u["civil_km"] + u["sistemas_km"]) + ESTACIONES_L1 * u["estacion"]
    return capex_total_2025("elevado_factibilidad_2017") - comp


@dataclass
class CostoLinea:
    tipo: str
    km: float
    estaciones: int
    civil: float
    sistemas: float
    estaciones_costo: float
    otros: float
    om_anual: float

    @property
    def capex(self) -> float:
        return self.civil + self.sistemas + self.estaciones_costo + self.otros


def costo_linea(tipo: str, km: float, estaciones: int, otros_por_km: float | None = None) -> CostoLinea:
    """CAPEX y O&M de una línea con los costos unitarios (billones de 2025).

    Los otros costos se escalan con la longitud a partir de los de la L1.
    """
    u = unitarios_2025(tipo)
    otros = (otros_costos_l1() / 23.96 if otros_por_km is None else otros_por_km) * km
    return CostoLinea(tipo, km, estaciones, km * u["civil_km"], km * u["sistemas_km"],
                      estaciones * u["estacion"], otros, km * u["om_km_anio"])


def comparar(km: float, estaciones: int) -> dict:
    """Comparación elevado vs. subterráneo para una misma línea."""
    e = costo_linea("elevado", km, estaciones)
    s = costo_linea("subterraneo", km, estaciones)
    return {"km": km, "estaciones": estaciones, "capex_elevado": e.capex, "capex_subterraneo": s.capex,
            "prima": s.capex / e.capex - 1.0, "ahorro_elevado": s.capex - e.capex,
            "om_elevado": e.om_anual, "om_subterraneo": s.om_anual}


def externalidad_umbral(diferencia_vp: float, anios, apertura: int, tasa: float, anio_vp: int) -> float:
    """Costo urbano anual del viaducto que iguala ambas tipologías.

    ``diferencia_vp`` = VP(subterráneo) - VP(elevado) sin externalidad. Si es
    positiva, el subterráneo es más costoso y la externalidad anual del
    viaducto que lo justificaría es ``diferencia_vp / sum(descuento)`` sobre
    los años de operación del viaducto.
    """
    import numpy as np

    anios = np.asarray(anios)
    activos = anios[anios >= apertura]
    suma = np.sum((1.0 + tasa) ** (-(activos - anio_vp).astype(float)))
    return float(diferencia_vp / suma) if suma > 0 else float("nan")
