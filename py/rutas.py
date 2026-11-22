"""Rutas del proyecto. Los scripts usan rutas relativas a su propia ubicación."""

import sys
from pathlib import Path

PY = Path(__file__).resolve().parent
RAIZ = PY.parent
DATOS = RAIZ / "datos"
SALIDAS = RAIZ / "salidas"
TEX = RAIZ / "tex"
FIGURAS = TEX / "figuras"
TABLAS = TEX / "tablas"

for d in (SALIDAS, FIGURAS, TABLAS):
    d.mkdir(parents=True, exist_ok=True)

if str(PY) not in sys.path:
    sys.path.insert(0, str(PY))
