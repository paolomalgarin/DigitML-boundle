@echo off

echo Stop di Apache...
taskkill /FI "WINDOWTITLE eq Apache - apache_start.bat"

echo Stop del python ML...
taskkill /FI "WINDOWTITLE eq Machine Learning - Start.bat"

echo Stop di Tomcat...
cd "xampp\tomcat\bin"
call "shutdown.bat"

echo Operazioni completate!
echo Puoi chiudere il terminale.
pause
