@echo off

echo Avvio di Apache...
cd "xampp"
start "Apache" /min "apache_start.bat"

echo Avvio di Tomcat...
cd "tomcat/bin"
start "" /min "startup.bat"

echo Avvio del python ML...
cd "../../../digitml-ML/bat_files"
start "Machine Learning" /min "Start.bat"

echo Apertura app...
start "" "http://127.0.0.1/"

echo Operazioni completate!
echo Puoi chiudere il terminale.
pause
