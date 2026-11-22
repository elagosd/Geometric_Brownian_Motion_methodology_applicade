@echo off
setlocal EnableExtensions

rem ===========================================================
rem  MODELO DE TRANSPORTE PARA BOGOTA
rem  Simulacion del diseno urbano optimo con Monte Carlo
rem  y movimiento browniano geometrico -- Eduardo Lagos Duran
rem
rem  Lanzador para Windows. Doble clic para abrir el menu.
rem
rem  NOTA: este archivo esta escrito sin tildes a proposito.
rem  La consola de Windows usa por defecto una pagina de
rem  codigos que no es UTF-8, y los acentos se verian como
rem  simbolos raros. El contenido con acentos esta en el
rem  documento y en README.md, no aqui.
rem ===========================================================

cd /d "%~dp0"
title Modelo de transporte para Bogota - Monte Carlo y MBG
color 0F

set "RAIZ=%~dp0"
set "VENV=%RAIZ%venv"
set "PY=%VENV%\Scripts\python.exe"

rem ---------- Localizar Python del sistema ----------
set "PYSIS="
py -3 --version >nul 2>&1 && set "PYSIS=py -3"
if not defined PYSIS (
  python --version >nul 2>&1 && set "PYSIS=python"
)

:MENU
cls
echo.
echo  ============================================================
echo    MODELO DE TRANSPORTE PARA BOGOTA
echo    Diseno urbano optimo con Monte Carlo y
echo    movimiento browniano geometrico
echo.
echo    Eduardo Lagos Duran
echo  ============================================================
echo.
echo    DOCUMENTO
echo      [1]  Abrir la investigacion en PDF
echo      [2]  Abrir el README
echo.
echo    APLICACION
echo      [3]  Abrir la aplicacion interactiva (Streamlit)
echo.
echo    REGENERAR
echo      [4]  Correr la simulacion y rehacer figuras y tablas
echo      [5]  Recompilar el PDF desde LaTeX
echo      [6]  Correr las pruebas del modelo
echo.
echo    OTROS
echo      [7]  Abrir la carpeta del proyecto
echo      [8]  Instalar / reparar el entorno de Python
echo      [9]  Diagnostico (que falta y que esta bien)
echo      [0]  Salir
echo.
if not exist "%VENV%\.dependencias_ok" (
  echo    [!] El entorno de Python aun no esta listo.
  echo        Las opciones 3, 4 y 6 lo instalaran automaticamente.
  echo.
)
set "OP="
set /p "OP=   Elija una opcion y pulse Enter:  "

if "%OP%"=="1" goto ABRIR_PDF
if "%OP%"=="2" goto ABRIR_README
if "%OP%"=="3" goto APLICACION
if "%OP%"=="4" goto REGENERAR
if "%OP%"=="5" goto COMPILAR
if "%OP%"=="6" goto PRUEBAS
if "%OP%"=="7" goto ABRIR_CARPETA
if "%OP%"=="8" goto INSTALAR
if "%OP%"=="9" goto DIAGNOSTICO
if "%OP%"=="0" goto FIN
goto MENU


rem ===========================================================
:ABRIR_PDF
if not exist "tex\investigacion.pdf" (
  echo.
  echo  [!] No se encuentra tex\investigacion.pdf
  echo      Use la opcion 5 para compilarlo.
  echo.
  pause
  goto MENU
)
start "" "tex\investigacion.pdf"
goto MENU


rem ===========================================================
:ABRIR_README
if exist "README.md" (
  start "" notepad "README.md"
) else (
  echo  [!] No se encuentra README.md
  pause
)
goto MENU


rem ===========================================================
:ABRIR_CARPETA
start "" "%RAIZ%"
goto MENU


rem ===========================================================
:APLICACION
call :ASEGURAR_ENTORNO
if errorlevel 1 goto MENU
cls
echo.
echo  ------------------------------------------------------------
echo   Iniciando la aplicacion interactiva...
echo.
echo   Se abrira en el navegador dentro de unos segundos.
echo   Si no se abre sola, entre a:   http://localhost:8501
echo.
echo   Para cerrarla: cierre la ventana negra que aparecera,
echo   o pulse Ctrl+C dentro de ella.
echo  ------------------------------------------------------------
echo.
start "Aplicacion - Transporte Bogota" /D "%RAIZ%" cmd /s /k ""%PY%" -m streamlit run app\app.py --server.headless true --browser.gatherUsageStats false"
timeout /t 12 /nobreak >nul
start "" "http://localhost:8501"
goto MENU


rem ===========================================================
:REGENERAR
call :ASEGURAR_ENTORNO
if errorlevel 1 goto MENU
cls
echo.
echo  ------------------------------------------------------------
echo   Regenerando todos los resultados del documento.
echo   La simulacion completa (fase II: 1.024 escenarios x 48 disenos;
echo   fase I: 2013) tarda alrededor de 30 minutos. No cierre esta ventana.
echo  ------------------------------------------------------------
echo.
pushd "%RAIZ%py"
echo  [1/4] Fase II: Monte Carlo, optimizacion y opciones reales...
"%PY%" correr_simulacion.py || goto REGEN_ERROR
echo        Fase I: coyuntura 2013 y metro elevado vs subterraneo...
"%PY%" correr_fase_2013.py || goto REGEN_ERROR
echo  [2/4] Figuras...
for %%F in (fig_01_red.py fig_02_mbg.py fig_03_convergencia.py fig_04_disenos.py fig_05_trayectorias.py fig_06_sensibilidad.py fig_07_opciones.py fig_08_fase2013.py) do (
  "%PY%" %%F || goto REGEN_ERROR
)
echo  [3/4] Cifras y tablas del documento...
"%PY%" generar_macros_tex.py || goto REGEN_ERROR
"%PY%" generar_tablas_tex.py || goto REGEN_ERROR
echo  [4/4] Copia de los resultados resumidos...
"%PY%" copiar_resultados.py || goto REGEN_ERROR
popd
echo.
echo  ------------------------------------------------------------
echo   Listo. Use la opcion 5 para recompilar el PDF.
echo  ------------------------------------------------------------
echo.
pause
goto MENU

:REGEN_ERROR
popd
echo.
echo  [!] Fallo uno de los pasos. Revise el mensaje de arriba.
echo.
pause
goto MENU


rem ===========================================================
:PRUEBAS
call :ASEGURAR_ENTORNO
if errorlevel 1 goto MENU
cls
"%PY%" -m pytest -q "%RAIZ%tests"
echo.
pause
goto MENU


rem ===========================================================
:COMPILAR
where latexmk >nul 2>&1
if errorlevel 1 (
  where pdflatex >nul 2>&1
  if errorlevel 1 (
    echo.
    echo  ------------------------------------------------------------
    echo   [!] No hay una distribucion de LaTeX instalada.
    echo.
    echo   Para compilar el documento necesita MiKTeX o TeX Live:
    echo       MiKTeX    https://miktex.org/download
    echo       TeX Live  https://tug.org/texlive/
    echo.
    echo   El PDF ya compilado esta en tex\investigacion.pdf.
    echo  ------------------------------------------------------------
    echo.
    pause
    goto MENU
  )
)
cls
pushd "%RAIZ%tex"
where latexmk >nul 2>&1
if errorlevel 1 (
  pdflatex -interaction=nonstopmode investigacion.tex
  bibtex investigacion
  pdflatex -interaction=nonstopmode investigacion.tex
  pdflatex -interaction=nonstopmode investigacion.tex
) else (
  latexmk -pdf -interaction=nonstopmode investigacion.tex
)
popd
if exist "tex\investigacion.pdf" (
  start "" "tex\investigacion.pdf"
) else (
  echo  [!] No se genero el PDF. Revise tex\investigacion.log
)
pause
goto MENU


rem ===========================================================
:INSTALAR
cls
echo.
echo   Instalando el entorno de Python desde cero.
echo.
if exist "%VENV%" (
  echo   Borrando el entorno anterior...
  rmdir /s /q "%VENV%"
)
call :ASEGURAR_ENTORNO
echo.
pause
goto MENU


rem ===========================================================
rem  Subrutina: asegura que el entorno virtual exista Y tenga las
rem  dependencias instaladas (archivo centinela). Devuelve
rem  errorlevel 1 si no lo consigue.
rem ===========================================================
:ASEGURAR_ENTORNO
set "CENTINELA=%VENV%\.dependencias_ok"
if exist "%PY%" if exist "%CENTINELA%" exit /b 0

if not defined PYSIS (
  echo.
  echo  ------------------------------------------------------------
  echo   [!] No se encontro Python en este equipo.
  echo.
  echo   Instalelo desde   https://www.python.org/downloads/
  echo   y marque la casilla "Add Python to PATH" durante la
  echo   instalacion. Luego vuelva a ejecutar este archivo.
  echo  ------------------------------------------------------------
  echo.
  pause
  exit /b 1
)

cls
echo.
echo  ------------------------------------------------------------
echo   Preparando el entorno de Python.
echo   Solo ocurre la primera vez. Tarda unos minutos.
echo   NO cierre esta ventana hasta que termine.
echo  ------------------------------------------------------------
echo.
echo   Python del sistema:
%PYSIS% --version
echo.

if not exist "%PY%" (
  echo   [1/3] Creando el entorno virtual...
  %PYSIS% -m venv "%VENV%"
) else (
  echo   [1/3] El entorno virtual ya existe, se reutiliza.
)
if not exist "%PY%" (
  echo.
  echo   [!] No se pudo crear el entorno virtual.
  echo       Compruebe que tiene permisos de escritura en esta carpeta.
  echo.
  pause
  exit /b 1
)

echo   [2/3] Instalando dependencias...
echo.
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r "%RAIZ%requirements.txt"

echo.
echo   [3/3] Verificando la instalacion...
"%PY%" -c "import streamlit, plotly, numpy, pandas, matplotlib, scipy" 2>nul
if errorlevel 1 (
  echo.
  echo  ------------------------------------------------------------
  echo   [!] Las dependencias NO quedaron instaladas.
  echo.
  echo   Detalle del fallo:
  "%PY%" -c "import streamlit, plotly, numpy, pandas, matplotlib, scipy"
  echo.
  echo   Causas mas frecuentes:
  echo.
  echo    - Sin conexion a internet.
  echo.
  echo    - Proxy o inspeccion SSL corporativa. Es lo mas probable si
  echo      esta en un equipo de empresa. Pruebe a instalar asi, desde
  echo      una consola abierta en esta misma carpeta:
  echo.
  echo        venv\Scripts\python.exe -m pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
  echo.
  echo    - Antivirus bloqueando la escritura en la carpeta venv.
  echo.
  echo   Cuando lo resuelva, vuelva a ejecutar la opcion 8.
  echo  ------------------------------------------------------------
  echo.
  pause
  exit /b 1
)

rem Centinela: marca que la instalacion se completo Y se verifico.
> "%CENTINELA%" echo instalacion verificada

echo.
echo   Entorno listo. Todas las dependencias verificadas.
timeout /t 3 /nobreak >nul
exit /b 0


rem ===========================================================
:DIAGNOSTICO
cls
echo.
echo  ------------------------------------------------------------
echo   DIAGNOSTICO DEL ENTORNO
echo  ------------------------------------------------------------
echo.
echo   Carpeta del proyecto:
echo     %RAIZ%
echo.
echo   Python del sistema:
if defined PYSIS (%PYSIS% --version) else (echo     [!] NO ENCONTRADO)
echo.
echo   Entorno virtual:
if exist "%PY%" (echo     OK    "%PY%") else (echo     [!] NO EXISTE)
if exist "%VENV%\.dependencias_ok" (echo     OK    dependencias verificadas) else (echo     [!] dependencias SIN verificar)
echo.
echo   Paquetes:
if exist "%PY%" (
  for %%P in (streamlit plotly numpy pandas matplotlib scipy pytest) do (
    "%PY%" -c "import %%P" >nul 2>&1 && (echo     OK    %%P) || (echo     FALTA %%P)
  )
) else (
  echo     [!] sin entorno virtual, no se puede comprobar
)
echo.
echo   LaTeX:
where latexmk >nul 2>&1 && (echo     OK    latexmk) || (echo     --    latexmk no encontrado)
where pdflatex >nul 2>&1 && (echo     OK    pdflatex) || (echo     --    pdflatex no encontrado)
echo.
echo   Archivos clave:
if exist "tex\investigacion.pdf" (echo     OK    tex\investigacion.pdf) else (echo     FALTA tex\investigacion.pdf)
if exist "app\app.py" (echo     OK    app\app.py) else (echo     FALTA app\app.py)
if exist "datos\calibracion.json" (echo     OK    datos\calibracion.json) else (echo     FALTA datos\calibracion.json)
if exist "requirements.txt" (echo     OK    requirements.txt) else (echo     FALTA requirements.txt)
echo.
echo  ------------------------------------------------------------
echo   Si falta algun paquete, use la opcion 8 para reinstalar.
echo  ------------------------------------------------------------
echo.
pause
goto MENU


rem ===========================================================
:FIN
endlocal
exit /b 0
