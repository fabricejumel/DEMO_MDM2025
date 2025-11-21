# DEMO_MDM2025

#INSTALLATION HTML
python3 -m venv venv_html
source venv_html/bin/activate 
pip install -r requirements_html.txt

#Lancement terminal 
cd html
./app.py

#Lancement navigateur  en mode kiosk
firefox --kiosk http://localhost:8000/
#quitter  navigateur 
Alt+F4
(Echap ou F11 ne marche pas)
