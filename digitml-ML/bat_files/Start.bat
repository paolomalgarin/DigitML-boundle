@echo off
cd ..

echo Attivazione dell'ambiente virtuale "DigitMl_ML"...
call DigitMl_ML\Scripts\activate

echo Avvio dell'app...
python app.py

echo Applicazione avviata. Premi CTRL+C per chiudere.
cmd /k