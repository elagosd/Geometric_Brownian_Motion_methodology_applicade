"""Copia a resultados/ los archivos resumidos de salidas/ (que no se versiona)."""

import shutil

import rutas

DESTINO = rutas.RAIZ / "resultados"
ARCHIVOS = ["resumen_disenos.csv", "ruta_optima.csv", "descuento.csv", "saltos.csv", "sensibilidad.csv",
            "sensibilidad_central.json", "convergencia.csv", "convergencia_ref.json", "opciones_reales.json",
            "calibracion_mbg.json", "calibracion_red.json", "trayectorias.json", "corrida.json",
            # Fase I (2013) y tipología del metro
            "fase2013_resumen.json", "fase2013_calibracion.json", "fase2013_exante.csv", "fase2013_expost.csv",
            "fase2013_descuento.csv", "fase2013_umbral.csv", "fase2_tipologia.csv", "metro_costos.csv"]


def main():
    DESTINO.mkdir(exist_ok=True)
    for nombre in ARCHIVOS:
        origen = rutas.SALIDAS / nombre
        if origen.exists():
            shutil.copy2(origen, DESTINO / nombre)
            print(f"  copiado: {nombre}")


if __name__ == "__main__":
    main()
