@echo off
SET VENV_NAME=DigitMl_ML

cd ..

echo Creazione dell'ambiente virtuale "%VENV_NAME%"...
python -m venv %VENV_NAME%

echo Attivazione dell'ambiente virtuale...
call %VENV_NAME%\Scripts\activate

echo Aggiornamento di pip...
python -m pip install --upgrade pip

echo Installazione delle dipendenze da requirements.txt...
pip install -r requirements.txt

deactivate

echo Setup completato! L'ambiente virtuale è attivo.
cmd /k
