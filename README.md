# DEMO_MDM2025

## Installation partie html/impression

```
python3 -m venv venv_html
source venv_html/bin/activate 
pip install -r requirements_html.txt
```

## Lancement terminal 

```
cd html
./app.py
```
## Lancement navigateur  en mode kiosk
```
firefox --kiosk http://localhost:8000/
```
## Quitter  navigateur en mode kiosk
Alt+F4
(Echap ou F11 ne marche pas)

> [!TIP]
> # Divers si besoin 
>
> ## Niveau encre
> ```
> hp-levels -i 
> ```
>
> ## Impressions réalisées
>
> ```
> lpstat -W completed 
> lpstat -W completed | wc -l
> ```

