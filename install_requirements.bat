@echo off
echo Installing required packages...
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Done! Setup complete.
echo To run the training, navigate to the project folder and run: python main.py
pause