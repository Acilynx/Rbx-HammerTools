@echo off
rem Opens the map converter window. Needs Python 3 from python.org (it includes tkinter).
where pyw >nul 2>nul && (start "" pyw -3 "%~dp0tools\converter_gui.py" & exit /b)
where pythonw >nul 2>nul && (start "" pythonw "%~dp0tools\converter_gui.py" & exit /b)
echo Python 3 was not found. Install it from https://www.python.org/downloads/ and run this again.
pause
