"""Parámetros del modelo y carga de los datos de entrada.

Cada bloque agrupa supuestos de una misma naturaleza. Los valores que
provienen de una fuente pública se documentan en ``FUENTES``; los demás son
supuestos del autor y se señalan como tales en el documento y en la
aplicación, donde pueden modificarse.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
DIR_DATOS = RAIZ / "datos"


@dataclass(frozen=True)
class Demanda:
    """Demanda motorizada en la hora pico y su dinámica estocástica."""

    # Viajes motorizados (transporte público + vehículo privado) en la hora
    # pico de la mañana. EM 2023: 14,64 millones de viajes/día en Bogotá y 20
    # municipios; 60 % en TP, auto, moto y taxi; factor de hora pico de 10 %.
    viajes_pico_base: float = 880_000.0
    anio_base: int = 2025
    anio_fin: int = 2050
    # Tendencia común de la tasa de viajes motorizados (supuesto del autor).
    mu_comun: float = 0.005
    # Volatilidad del factor común y carga de cada zona sobre él.
    sigma_comun: float = 0.05
    ocupacion_auto: float = 1.30
    # Saltos sistémicos tipo Merton (desactivados en el caso base).
    intensidad_saltos: float = 0.0
    media_salto: float = -0.35
    desv_salto: float = 0.10
    compensar_saltos: bool = True


@dataclass(frozen=True)
class Red:
    """Oferta vial y de transporte público."""

    vel_libre_via: float = 40.0  # km/h
    bpr_alfa: float = 0.15  # Bureau of Public Roads (1964)
    bpr_beta: float = 4.0
    hac_alfa: float = 0.6  # penalidad por hacinamiento en el TP
    hac_beta: float = 4.0
    # Más allá de x_lim la función se extiende por su tangente y se acota en
    # f_max: representa la dispersión de la hora pico (los viajes se corren a
    # las horas vecinas) y el abordaje restringido en el TP.
    bpr_x_lim: float = 2.0
    vel_min_via: float = 4.0  # km/h
    hac_x_lim: float = 1.5
    hac_f_max: float = 4.0
    vel_ref_via: float = 20.0  # velocidades de referencia para accesibilidad
    vel_ref_zonal: float = 15.0  # bus zonal (SITP 12–14 km/h; alimentadores 17,5)
    espera_acceso_tp: float = 15.0  # min
    transbordo: float = 6.0  # min por transbordo
    acceso_auto: float = 5.0  # min
    intrazonal_auto: float = 12.0  # min
    intrazonal_tp: float = 18.0  # min
    circuidad: float = 1.30
    lambda_logit: float = 0.05  # 1/min
    iter_msa: int = 60
    paso_inicial: float = 0.5
    paso_minimo: float = 0.02
    tol_equilibrio: float = 2e-4
    # Calibrados en el año base (ver calibracion.py).
    escala_cap_via: float = 8000.0
    asc_tp: float = 0.0
    beta_gravedad: float = 0.04


@dataclass(frozen=True)
class Economia:
    """Valoración económica (precios constantes de 2026)."""

    vtts_cop_h: float = 8000.0  # valor del tiempo de viaje
    factor_anual: float = 3000.0  # horas pico equivalentes por año
    tasa_descuento: float = 0.09  # DNP, Resolución 1092 de 2022
    anio_vp: int = 2026
    costo_oper_auto_km: float = 700.0  # COP por vehículo-km
    externalidad_auto_km: float = 350.0  # emisiones, ruido y siniestralidad
    costo_oper_bus_paxkm: float = 180.0  # COP por pasajero-km en bus
    om_ferro_frac: float = 0.025  # O&M anual como fracción del CAPEX
    vida_util: int = 50
    tarifa_tp: float = 3200.0  # COP (percepción del usuario)
    costo_monetario_auto_km: float = 450.0
    costo_fijo_auto: float = 3500.0
    co2_kg_vehkm: float = 0.20


@dataclass(frozen=True)
class Sobrecostos:
    """Sobrecostos de CAPEX (Flyvbjerg, 2005; Banco Mundial WP 3781)."""

    activar: bool = True
    ferro_media: float = 1.447  # 44,7 % de sobrecosto medio, desv. 38,4 %
    ferro_desv: float = 0.384
    brt_media: float = 1.204  # vías: 20,4 %, desv. 29,9 %
    brt_desv: float = 0.299


@dataclass(frozen=True)
class UsoSuelo:
    """Política de proximidad y policentrismo (POT, Decreto 555 de 2021)."""

    reduccion_motorizada_max: float = 0.05  # viajes que pasan a pie o bici
    relocalizacion_centro_max: float = 0.15  # fracción de la atracción del centro
    costo_anual_billones: float = 0.6  # con intensidad 1
    tiempo_no_motorizado: float = 20.0  # min de cada viaje que pasa a pie o bici
    anio_inicio: int = 2027
    anio_pleno: int = 2040


@dataclass(frozen=True)
class Metas:
    """Metas de calibración del año base."""

    vel_via: float = 18.9  # TomTom Traffic Index 2025, hora pico
    part_tp: float = 35.0 / 60.0  # EM 2023: TP 35 %, auto+moto+taxi 25 %
    tiempo_medio: float = 48.0  # EM 2023: duración promedio de los viajes


@dataclass
class Parametros:
    demanda: Demanda = field(default_factory=Demanda)
    red: Red = field(default_factory=Red)
    economia: Economia = field(default_factory=Economia)
    sobrecostos: Sobrecostos = field(default_factory=Sobrecostos)
    suelo: UsoSuelo = field(default_factory=UsoSuelo)
    metas: Metas = field(default_factory=Metas)
    zonas: pd.DataFrame = field(default_factory=lambda: cargar_zonas())
    corredores: pd.DataFrame = field(default_factory=lambda: cargar_corredores())
    proyectos: pd.DataFrame = field(default_factory=lambda: cargar_proyectos())

    # ------------------------------------------------------------------
    @property
    def n_zonas(self) -> int:
        return len(self.zonas)

    @property
    def anios(self) -> np.ndarray:
        return np.arange(self.demanda.anio_base, self.demanda.anio_fin + 1)

    @property
    def mu_total(self) -> np.ndarray:
        """Deriva anual de la demanda por zona: demografía + tasa común."""
        return self.zonas["mu_demografica"].to_numpy(float) + self.demanda.mu_comun

    @property
    def participacion_generacion(self) -> np.ndarray:
        z = self.zonas
        peso = z["poblacion_miles"].to_numpy(float) * z["tasa_generacion"].to_numpy(float)
        return peso / peso.sum()

    @property
    def proyectos_decision(self) -> list[str]:
        p = self.proyectos
        return p.loc[p["estado"] == "decision", "id"].tolist()

    @property
    def proyectos_comprometidos(self) -> list[str]:
        p = self.proyectos
        return p.loc[p["estado"] == "comprometido", "id"].tolist()

    def con(self, **cambios) -> "Parametros":
        """Copia con bloques reemplazados, p. ej. ``p.con(red=replace(p.red, ...))``."""
        return replace(self, **cambios)

    def ajustar(self, bloque: str, **valores) -> "Parametros":
        """Copia con algunos valores de un bloque cambiados."""
        actual = getattr(self, bloque)
        return replace(self, **{bloque: replace(actual, **valores)})


# ----------------------------------------------------------------------
def cargar_zonas(ruta: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(ruta or DIR_DATOS / "zonas.csv")


def cargar_corredores(ruta: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(ruta or DIR_DATOS / "corredores.csv")
    df["vel_tp_base"] = pd.to_numeric(df["vel_tp_base"], errors="coerce")
    return df


def cargar_proyectos(ruta: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(ruta or DIR_DATOS / "proyectos.csv")


def cargar_serie_transmilenio(ruta: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(ruta or DIR_DATOS / "calibracion_transmilenio.csv")


def parametros_calibrados() -> Parametros:
    """Parámetros con el año base calibrado (usa el archivo de calibración si existe)."""
    from .calibracion import cargar_o_calibrar

    return cargar_o_calibrar(Parametros())


FUENTES = {
    "viajes_dia": "Secretaría Distrital de Movilidad, Encuesta de Movilidad 2023 Bogotá-Región",
    "reparto_modal": "EM 2023: TP 35 %, a pie 28 %, auto 14 %, bici 7 %, moto 7 %, taxi 4 %",
    "duracion": "EM 2023: 48 minutos de duración promedio",
    "velocidad": "TomTom Traffic Index 2025: 18,9 km/h y séptimo lugar mundial en congestión",
    "poblacion": "DANE, proyecciones post-COVID (2025): 7.937.898 habitantes en Bogotá",
    "tasa_descuento": "DNP, Resolución 1092 de 2022: tasa social de descuento de 9 %",
    "sobrecostos": "Flyvbjerg (2005), Banco Mundial WP 3781",
    "metro_l1": "Empresa Metro de Bogotá: 72.000 pasajeros/hora-sentido; 43 km/h; marzo de 2028",
    "troncal_vel": "Velocidad comercial de troncales TransMilenio 2024: 25,2 km/h",
}
