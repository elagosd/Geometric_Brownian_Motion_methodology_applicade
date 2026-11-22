# Guía: de la investigación al código

Este documento relaciona cada parte de la investigación (`tex/investigacion.pdf`) con el código Python que la produce.

| Capítulo | Contenido | Código | Salidas |
|---|---|---|---|
| 3–5 | MBG multivariado, saltos, calibración y convergencia | `py/modelo/gbm.py` | `salidas/calibracion_mbg.json`, `escenarios.npz`, `convergencia.csv` |
| 4 | Red, demanda, equilibrio y calibración 2025 | `red.py`, `demanda.py`, `asignacion.py`, `calibracion.py` | `datos/calibracion.json` |
| 4 | Costo social y valor presente | `evaluacion.py` | `vp_disenos.npz` |
| 6 | Diseño óptimo media–CVaR | `optimizacion.py`, `riesgo.py` | `resumen_disenos.csv`, `ruta_optima.csv` |
| 7–8 | **Fase I**: coyuntura y evaluación de 2013 | `fases.py`, `py/correr_fase_2013.py` | `fase2013_*.csv/json/npz` |
| 9 | **Fase II**: resultados 2025 | `py/correr_simulacion.py` | `resumen_disenos.csv`, `descuento.csv`, `saltos.csv`, `sensibilidad.csv` |
| 10 | Línea 3 como opción real | `opciones_reales.py` | `opciones_reales.json` |
| 11 | Metro elevado vs. subterráneo | `metro.py`, `py/correr_fase_2013.py` | `metro_costos.csv`, `fase2013_umbral.csv`, `fase2_tipologia.csv` |

## Flujo de una simulación

1. `parametros_calibrados()` carga datos y calibración (`parametros.py`).
2. `gbm.escenarios_desde_parametros()` simula los índices de demanda por zona (Sobol + solución exacta).
3. `evaluar(p, diseno, escenarios, sobrecostos)` recorre los años: construye la oferta, calcula la matriz OD, resuelve el equilibrio y acumula el costo social.
4. `resumen()` calcula media, CVaR, beneficio neto y frontera de Pareto.
5. `generar_macros_tex.py` y `generar_tablas_tex.py` llevan las cifras al documento.

## Fase I en detalle

- `fases.parametros_2013()`: demanda base de 2013 (EM 2011), población según las proyecciones del censo de 2005, derivas vistas desde 2013, tasa de 12 % y VP a 2014.
- `fases.calibrar_2013()`: ajusta la escala de capacidad vial y la constante modal a la velocidad y la participación del transporte público de 2013.
- `fases.proyectos_2013()`: elevado (L1E), subterráneo con costo oficial (L1S) y con costos unitarios (L1S_alto).
- `fases.escenarios_expost()`: trayectoria observada 2013–2025 seguida de los escenarios de la fase II.
- `metro.externalidad_umbral()`: costo urbano anual del viaducto que iguala ambas tipologías.

## Cambiar un supuesto

- Datos: `datos/*.csv`.
- Parámetros económicos y de red: `py/modelo/parametros.py` (o en la barra lateral de la aplicación).
- Costos del metro: `py/modelo/metro.py`.
- Después de un cambio, ejecute la opción «Regenerar» de `INICIAR.bat`, o los pasos del README.
