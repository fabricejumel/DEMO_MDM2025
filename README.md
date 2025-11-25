# DEMO Mondiale des metiers 2025 

> [!WARNING]  
> Verifier la commande d'impression finale  avec les noms datés 


## PARTIE IMPRESSION 


### Installation partie html/impression

```
python3 -m venv venv_html
source venv_html/bin/activate 
pip install -r requirements_html.txt
```

### Lancement terminal 

```
cd html
./app.py
```
### Lancement navigateur  en mode kiosk
```
firefox --kiosk http://localhost:8000/
```
### Quitter  navigateur en mode kiosk
Alt+F4
(Echap ou F11 ne marche pas)

> [!TIP]
> ## Divers si besoin 
>
> ### Niveau encre
> ```
> hp-levels -i 
> ```
>
> ### Impressions réalisées
>
> ```
> lpstat -W completed 
> lpstat -W completed | wc -l
> ```

## PARTIE GENETATION

### Installation partie html/impression

```
python3 -m venv_IA_Stablediffusion 
source venv_IA_Stablediffusion/bin/activate 
pip install -r requirements_Stablediffusion.txt
```
### test Stable Diffusion dans venv_IA_Stablediffusion

```
python test_stablediffusion.py `

```


