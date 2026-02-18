@echo off
echo Installing dependencies for ComfyUI_Goutam_Nano_Suite...

:: Try to find the embedded Python (standard ComfyUI Portable location)
if exist "..\..\..\python_embeded\python.exe" (
    echo Found embedded Python.
    ..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
) else (
    echo Embedded Python not found. Trying system Python...
    python -m pip install -r requirements.txt
)

echo.
echo Installation complete. Please restart ComfyUI.
pause
