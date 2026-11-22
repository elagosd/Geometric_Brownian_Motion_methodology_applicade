"""Costo social de un diseño urbano por año y por escenario.

Un diseño es un subconjunto de los proyectos en decisión más una intensidad
``theta`` de la política de proximidad y policentrismo. Para cada escenario
del MBG y cada año se calcula la matriz OD, el equilibrio modal y el costo
social (billones de COP de 2026):

    tiempo      VTTS x horas-persona x factor anual
    auto        (costo operativo + externalidades) x veh-km x factor anual
    bus         costo operativo x pasajero-km en bus x factor anual
    O&M         fracción anual del CAPEX de los proyectos férreos en operación
    suelo       costo anual de la política de proximidad
    CAPEX       inversión repartida en los años de construcción, con sobrecosto
    residual    valor remanente en 2050 (depreciación lineal), con signo negativo

El valor presente usa la tasa social de descuento a partir de 2026. Los
proyectos comprometidos (Línea 1, troncal Av. 68, Regiotram de Occidente) son
comunes a todos los diseños y su CAPEX no entra en la comparación.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .asignacion import equilibrio
from .demanda import matriz_od, rampa
from .red import abiertos_en, construir_oferta, huella

COMPONENTES = ("tiempo", "auto", "bus", "om", "suelo", "capex", "residual")
BILLON = 1e12


@dataclass(frozen=True)
class Diseno:
    proyectos: tuple[str, ...] = ()
    theta: float = 0.0
    aperturas: tuple[tuple[str, int], ...] = ()

    @property
    def nombre(self) -> str:
        base = "+".join(self.proyectos) if self.proyectos else "Base"
        return f"{base} | θ={self.theta:g}"

    @property
    def dic_aperturas(self) -> dict[str, int]:
        return dict(self.aperturas)


@dataclass
class ResultadoDiseno:
    diseno: Diseno
    anios: np.ndarray  # (Y,)
    componentes: dict[str, np.ndarray]  # nombre -> (n, Y)
    descuento: np.ndarray  # (Y,)
    tiempo_medio: np.ndarray  # (n, Y)
    part_tp: np.ndarray
    vel_via: np.ndarray
    hac_max: np.ndarray
    co2_mt: np.ndarray  # millones de toneladas por año
    capex_total: np.ndarray = field(default_factory=lambda: np.zeros(1))  # (n,)

    @property
    def costo_anual(self) -> np.ndarray:
        return sum(self.componentes.values())

    @property
    def vp(self) -> np.ndarray:
        """Valor presente del costo social por escenario (billones de COP)."""
        return self.costo_anual @ self.descuento

    def vp_componente(self, nombre: str) -> np.ndarray:
        return self.componentes[nombre] @ self.descuento


def muestrear_sobrecostos(p, n: int, semilla: int | None = 11) -> dict[str, np.ndarray]:
    """Multiplicadores lognormales de CAPEX por escenario y proyecto.

    La media y la desviación vienen de Flyvbjerg (2005) por tipo de proyecto.
    Se usan los mismos multiplicadores en todos los diseños (números
    aleatorios comunes), de modo que las comparaciones no mezclan ruido.
    """
    s = p.sobrecostos
    rng = np.random.default_rng(semilla)
    salida = {}
    for _, fila in p.proyectos.iterrows():
        if not s.activar:
            salida[fila["id"]] = np.ones(n)
            continue
        m, d = (s.ferro_media, s.ferro_desv) if fila["clase_sobrecosto"] == "ferro" else (s.brt_media, s.brt_desv)
        s2 = np.log(1.0 + (d / m) ** 2)
        salida[fila["id"]] = rng.lognormal(np.log(m) - 0.5 * s2, np.sqrt(s2), size=n)
    return salida


def factores_descuento(p, anios: np.ndarray) -> np.ndarray:
    e = p.economia
    return (1.0 + e.tasa_descuento) ** (-(anios - e.anio_vp).astype(float))


def flujos_inversion(p, diseno: Diseno, anios: np.ndarray, sobrecostos: dict[str, np.ndarray],
                     n: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """CAPEX, O&M, residual y costo de suelo por año, forma ``(n, Y)``."""
    e, s = p.economia, p.suelo
    pr = p.proyectos.set_index("id")
    Y = anios.size
    capex = np.zeros((n, Y))
    om = np.zeros((n, Y))
    residual = np.zeros((n, Y))
    capex_total = np.zeros(n)
    ap = diseno.dic_aperturas
    for pid in diseno.proyectos:
        fila = pr.loc[pid]
        apertura = int(ap.get(pid, fila["apertura"]))
        n_c = int(fila["anios_construccion"])
        inicio = max(apertura - n_c, int(anios[0]))
        anios_c = np.arange(inicio, apertura)
        if anios_c.size == 0:
            anios_c = np.array([int(anios[0])])
        monto = fila["capex_billones"] * sobrecostos[pid]  # (n,)
        capex_total += monto
        for a in anios_c:
            k = np.searchsorted(anios, a)
            if k < Y and anios[k] == a:
                capex[:, k] += monto / anios_c.size
        abiertos = anios >= apertura
        om[:, abiertos] += e.om_ferro_frac * fila["capex_billones"]
        vida_usada = max(0, int(anios[-1]) - apertura + 1)
        resto = max(0.0, 1.0 - vida_usada / e.vida_util)
        residual[:, -1] -= monto * resto
    suelo = np.zeros((n, Y))
    if diseno.theta > 0:
        activo = (anios >= s.anio_inicio) & (anios <= s.anio_pleno)
        suelo[:, activo] = diseno.theta * s.costo_anual_billones
    return {"capex": capex, "om": om, "residual": residual, "suelo": suelo}, capex_total


def evaluar(p, diseno: Diseno, escenarios, sobrecostos: dict[str, np.ndarray],
            paso: int = 1, anio_inicio: int = 2026) -> ResultadoDiseno:
    """Evalúa un diseño en todos los escenarios.

    ``paso > 1`` evalúa el equilibrio cada ``paso`` años (más el último) e
    interpola linealmente los componentes operativos; útil en la aplicación.
    """
    e, d = p.economia, p.demanda
    anios_todos = escenarios.anios
    sel = anios_todos >= anio_inicio
    anios = anios_todos[sel]
    X = escenarios.X[:, sel, :]
    n, Y = X.shape[0], anios.size
    eval_idx = list(range(0, Y, max(1, paso)))
    if eval_idx[-1] != Y - 1:
        eval_idx.append(Y - 1)

    clave = huella(p)
    oper = {k: np.full((n, len(eval_idx)), np.nan) for k in ("tiempo", "auto", "bus")}
    ind = {k: np.zeros((n, len(eval_idx))) for k in ("tiempo_medio", "part_tp", "vel_via", "hac_max", "co2")}
    s_prev = None
    abiertos_prev = None
    for m, k in enumerate(eval_idx):
        anio = int(anios[k])
        abiertos = abiertos_en(p, anio, diseno.proyectos, diseno.dic_aperturas)
        oferta = construir_oferta(p, abiertos, clave)
        # La distribución de destinos usa la red comprometida en todos los
        # diseños: con demanda total fija, el costo de recursos compara
        # diseños sobre la misma matriz OD (ver el capítulo del modelo).
        oferta_ref = construir_oferta(p, abiertos_en(p, anio, ()), clave)
        T = matriz_od(p, oferta_ref, X[:, k, :], anio, diseno.theta)
        s0 = s_prev if abiertos == abiertos_prev else None
        res = equilibrio(p, oferta, T, s0=s0)
        s_prev, abiertos_prev = res.s_tp, abiertos
        F = e.factor_anual
        # Viajes que la política de proximidad convierte a modos no motorizados.
        su = p.suelo
        frac = su.reduccion_motorizada_max * diseno.theta * rampa(anio, su.anio_inicio, su.anio_pleno)
        nm = res.viajes * frac / max(1.0 - frac, 1e-12)
        horas_nm = nm * su.tiempo_no_motorizado / 60.0
        oper["tiempo"][:, m] = e.vtts_cop_h * (res.horas_persona + horas_nm) * F / BILLON
        oper["auto"][:, m] = (e.costo_oper_auto_km + e.externalidad_auto_km) * res.veh_km * F / BILLON
        oper["bus"][:, m] = e.costo_oper_bus_paxkm * res.pax_km_bus * F / BILLON
        ind["tiempo_medio"][:, m] = res.tiempo_medio
        ind["part_tp"][:, m] = res.participacion_tp
        ind["vel_via"][:, m] = res.velocidad_media_via(oferta.longitud)
        ind["hac_max"][:, m] = res.hacinamiento.max(axis=1)
        ind["co2"][:, m] = e.co2_kg_vehkm * res.veh_km * F / 1e9  # Mt

    def interpolar(a):
        if len(eval_idx) == Y:
            return a
        xs = anios[eval_idx]
        return np.stack([np.interp(anios, xs, fila) for fila in a])

    comp = {k: interpolar(v) for k, v in oper.items()}
    inversion, capex_total = flujos_inversion(p, diseno, anios, sobrecostos, n)
    comp.update(inversion)
    comp = {k: comp[k] for k in COMPONENTES}
    return ResultadoDiseno(
        diseno=diseno, anios=anios, componentes=comp, descuento=factores_descuento(p, anios),
        tiempo_medio=interpolar(ind["tiempo_medio"]), part_tp=interpolar(ind["part_tp"]),
        vel_via=interpolar(ind["vel_via"]), hac_max=interpolar(ind["hac_max"]),
        co2_mt=interpolar(ind["co2"]), capex_total=capex_total)
