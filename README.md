# Modelo de transporte para Bogotá: Monte Carlo y movimiento browniano geométrico

**Transportation Model for Bogotá — Simulating the optimal urban design with Monte Carlo and the Geometric Brownian Motion methodology**

Autor: Eduardo Lagos Durán

Esta investigación modela la movilidad de Bogotá y su región con una demanda estocástica. El índice de demanda de doce zonas sigue un movimiento browniano geométrico (MBG) multivariado, con saltos de tipo Merton como extensión. Sobre esa demanda se busca, con simulación de Monte Carlo, el diseño urbano óptimo: qué proyectos férreos construir y con qué intensidad aplicar la política de suelo. El trabajo tiene dos fases:

| Fase | Periodo | Pregunta |
|---|---|---|
| **I** | Coyuntura de 2013 | ¿Qué decisión de primera línea de metro era la mejor con la información de entonces? ¿Y con lo que realmente ocurrió? |
| **II** | Condiciones de 2025 | ¿Qué combinación de proyectos y de política de suelo minimiza el costo social entre 2026 y 2050? |

Las dos fases se combinan en un capítulo de síntesis sobre si **la Primera Línea del Metro, que se construye elevada, debió ser subterránea dado el tamaño de Bogotá**.

Todas las simulaciones están escritas en **Python** y vienen en este repositorio junto con sus resultados (`salidas/`).

---

## Resultados principales

### Fase I · 2013 (tasa de descuento de 12 %, vigente entonces)

| Alternativa de primera línea | E[BN] al 12 % | al 9 % | al 6,4 % |
|---|---:|---:|---:|
| Elevado (Bosa–calle 72) | **4,6** | 15,5 | 32,6 |
| Subterráneo, costo oficial 2014 | 1,1 | 14,1 | **36,0** |
| Subterráneo, costos unitarios 2023 | −17,4 | −7,2 | 11,8 |

Billones de pesos de 2025, valor presente a 2014, frente a no construir metro.

- La demanda observada en 2025 quedó en el **percentil 2,4** de los escenarios construidos con la información de 2013 (menos población según el censo de 2018, pandemia y recuperación incompleta).
- *Ex post*, con la demanda real, el subterráneo habría perdido más que el elevado con cualquier fecha de apertura.

### Fase II · 2025 (tasa de 9 %)

- **Diseño óptimo:** Regiotram del Norte con la política de proximidad completa (`RTN | θ=1`). Su beneficio neto esperado es de 16,6 billones y es positivo en el 98 % de los escenarios.
- Las líneas de metro adicionales reducen la dispersión del costo, pero no se pagan a esa tasa. Con saltos sistémicos, la combinación con la Línea 3 tiene el menor CVaR.
- Línea 3 como opción real (mínimos cuadrados de Monte Carlo): comprometerse hoy tiene un VPN esperado de −19,2 billones; la regla óptima es esperar.

### ¿Elevado o subterráneo?

1. Con la información y la tasa de 2013, **el elevado tenía el mayor beneficio social esperado**.
2. **La conclusión depende de la tasa de descuento.** Con 9 % y el costo oficial de 2014 las dos tipologías empatan, y por debajo de **~8,2 %** el subterráneo de 2014 sería mejor.
3. **El tamaño de Bogotá, por sí solo, no justifica el subterráneo.** La capacidad del viaducto no es la restricción, y con un kilómetro subterráneo se pagan **1,73 km elevados**: una ciudad grande necesita una red.
4. **El argumento decisivo es el costo urbano del viaducto.** Para inclinar la decisión, al 12 % tendría que superar **0,88 billones/año**; al 9 % basta con **0,23**. Una estimación por pérdida de valor de los predios da **0,06–0,28 billones/año**.
5. **Líneas 2 y 3:** como subterráneas cuestan 10–17 billones más que una variante elevada. Una Línea 3 elevada entraría en el diseño óptimo.

El razonamiento completo está en el capítulo 11 del documento.

---

## Estructura del repositorio

```
datos/            zonas, corredores, proyectos, serie de TransMilenio, calibraciones 2013 y 2025
py/modelo/        paquete del modelo
  gbm.py            MBG multivariado exacto, saltos de Merton, Sobol, calibración MLE
  red.py            red estilizada, caminos mínimos, oferta por año
  demanda.py        generación, atracción, modelo gravitacional (Furness)
  asignacion.py     equilibrio modal con congestión BPR y hacinamiento
  calibracion.py    calibración del año base 2025
  evaluacion.py     costo social, sobrecostos lognormales, valor presente
  riesgo.py         VaR, CVaR (Rockafellar–Uryasev), frontera de Pareto
  optimizacion.py   enumeración de diseños, criterio media–CVaR
  opciones_reales.py  LSM (Longstaff–Schwartz) y McDonald–Siegel
  fases.py          fase I: parámetros, calibración y escenarios de 2013
  metro.py          costos unitarios elevado/subterráneo, costo urbano umbral
py/correr_simulacion.py   simulación de la fase II
py/correr_fase_2013.py    simulación de la fase I y de las tipologías del metro
py/fig_0*.py              figuras del documento
py/generar_*.py           cifras y tablas de LaTeX a partir de las salidas
salidas/          resultados completos de las simulaciones (CSV, JSON, NPZ)
resultados/       copia resumida de los resultados
app/              aplicación Streamlit
tests/            pruebas automáticas (pytest)
tex/              documento LaTeX (investigacion.pdf)
docs/             guía para recorrer el código
INICIAR.bat       menú para Windows
```

## Cómo usarlo

### Windows

Doble clic en `INICIAR.bat`. El menú permite abrir el PDF, lanzar la aplicación, regenerar los resultados, compilar el documento, correr las pruebas e instalar las dependencias.

### Línea de comandos

```bash
pip install -r requirements.txt

# Aplicación interactiva
streamlit run app/app.py

# Reproducir todos los resultados
cd py
python correr_simulacion.py      # fase II: 1.024 escenarios x 48 diseños (~25 min)
python correr_fase_2013.py       # fase I y tipologías del metro (~3 min)
for f in fig_0*.py; do python $f; done
python generar_macros_tex.py && python generar_tablas_tex.py && python copiar_resultados.py

# Documento
cd ../tex && latexmk -pdf investigacion.tex

# Pruebas
cd .. && pytest -q tests
```

Las cifras del documento no se escriben a mano: `generar_macros_tex.py` las lee de `salidas/`. Las semillas son fijas (2026 para la fase II y 2013 para la fase I).

## Aplicación

| Página | Contenido |
|---|---|
| Inicio | Resumen y supuestos |
| Demanda (MBG) | Calibración, abanico de escenarios y saltos |
| Evaluar un diseño | Costo social, trayectorias y riesgo de cualquier combinación |
| Diseño óptimo | Los 48 diseños, frontera media–CVaR y elección según el peso del riesgo |
| Opciones reales | Regla de ejercicio LSM y validación con McDonald–Siegel |
| Fase I · 2013 | Escenarios vistos desde 2013, evaluación *ex ante* y *ex post* |
| Metro: elevado vs. subterráneo | Mapa de decisión interactivo según el costo urbano del viaducto y la tasa |
| Convergencia MC | Monte Carlo, variables antitéticas y Sobol |
| Metodología | Ecuaciones y fuentes |

## Fuentes principales

Encuesta de Movilidad 2011, 2019 y 2023 (SDM); DANE (cuentas departamentales, censo 2018); Banco de la República (ICER 2013); Empresa Metro de Bogotá (diseño 2014, estudio de subterranización de 2023, líneas 2 y 3); CONPES 3900; TomTom Traffic Index; Catastro Distrital (censo inmobiliario 2025); Flyvbjerg et al. (sobrecostos y demanda); Ahlfeldt, Nitsch y Wendland (2016, 2019) y Strand y Vågnes (2001) sobre el costo urbano de los viaductos. La bibliografía completa está en `tex/bibliografia.bib`.

## Limitaciones

Es un modelo **estratégico** (12 zonas, 18 corredores): sirve para comparar diseños con los mismos escenarios, no para dimensionar estaciones. El costo urbano del viaducto se trata con umbrales y una estimación externa. La trayectoria observada 2013–2025 es estilizada. Los detalles están en el capítulo 12.
