"""Modelo estocástico de transporte y diseño urbano para Bogotá.

Paquete de soporte de la investigación «Modelo de transporte para Bogotá:
simulación del diseño urbano óptimo con Monte Carlo y movimiento browniano
geométrico». Los módulos se organizan así:

    parametros      Parámetros, datos de entrada y sus fuentes.
    gbm             Simulación exacta del MBG multivariado, calibración y saltos.
    red             Grafo de corredores, caminos mínimos e incidencias.
    demanda         Generación de viajes, política de suelo y modelo gravitacional.
    asignacion      Partición modal logit con congestión (BPR) y hacinamiento.
    evaluacion      Costo social por año y valor presente por escenario.
    riesgo          VaR, CVaR, intervalos bootstrap y frontera de Pareto.
    optimizacion    Enumeración de diseños y criterios de decisión.
    opciones_reales Mínimos cuadrados de Monte Carlo y validación analítica.
    calibracion     Ajuste del año base a las metas observadas.
"""

__version__ = "1.0.0"
