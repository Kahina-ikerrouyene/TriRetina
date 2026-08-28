@echo off
title TriRetina - Lancement
color 0A

echo.
echo  TriRetina - Systeme d'analyse retinienne
echo  ==========================================
echo.

echo  [1/2] Verification de Python 3.11...
py -3.11 --version
if errorlevel 1 (
    echo.
    echo  [ERREUR] Python 3.11 est introuvable.
    echo.
    pause
    exit /b 1
)

echo.
echo  [2/2] Verification des dependances...
py -3.11 -m pip install -r requirements.txt --disable-pip-version-check

if errorlevel 1 (
    echo.
    echo  [ERREUR] Echec de l'installation des dependances.
    echo.
    pause
    exit /b 1
)

echo.
echo  Demarrage de l'application TriRetina...
echo.
echo  L'application va s'ouvrir dans votre navigateur.
echo  Pour arreter : Ctrl+C ou fermez cette fenetre.
echo.

py -3.11 -m streamlit run app.py --server.headless false

pause