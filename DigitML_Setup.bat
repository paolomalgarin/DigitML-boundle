@echo off

echo Setup di Apache e Tomcat...
cd "xampp"
echo %CD%
start "Xampp Setup" "setup_xampp.bat"

cd "../"

echo Setup del python ML...
cd "digitml-ML\bat_files"
echo %CD%
start "Machine Learning Setup" "Setup.bat"
echo Setup completato
echo Puoi chiudere il terminale.
pause