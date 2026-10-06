# TriRetina — Détection de Pathologies Rétiniennes

Application de télé-ophtalmologie basée sur le deep learning pour la détection automatique de trois pathologies rétiniennes à partir de photographies du fond d'œil (rétinographies).

## Interface web

<p align="center">
  <img src="TriRetina_GitHub/2_Application_TriRetina/streamlit_interface.png" alt="Interface web TriRetina" width="900">
</p>

## Résultats obtenus

| Pathologie              |        AUC | Seuil optimal |
| ----------------------- | ---------: | ------------: |
| Rétinopathie Diabétique | **0.9186** |          40 % |
| Glaucome                | **0.9266** |          35 % |
| DMLA                    | **0.9175** |          45 % |
| **Moyenne**             | **0.9209** |             — |

## Modèle

* **EfficientNetB3** pré-entraîné sur ImageNet avec fine-tuning bout-en-bout
* Classification binaire indépendante pour chaque pathologie
* Fonction d'activation **sigmoid**
* Seuils optimisés à partir des courbes ROC avec maximisation du **Youden Index**
* Score affiché normalisé pour faciliter l'interprétation des résultats

## Technologies

**Python · TensorFlow / Keras · Streamlit · OpenCV · NumPy · fpdf2**

## Dataset

Environ **10 000 images de fond d'œil annotées** (rétinographies couleur).

Sources :

* APTOS 2019
* REFUGE
* ORIGA
* DRISHTI
* Kaggle EyePACS

## Structure du projet

```text
TriRetina/
├── README.md
│
└── TriRetina_GitHub/
    ├── 1_Notebook_Entrainement/
    │   └── # Entraînement et évaluation du modèle EfficientNetB3
    │
    └── 2_Application_TriRetina/
        ├── Images test/
        ├── LANCER.bat
        ├── app.py
        ├── pyproject.toml
        ├── requirements.txt
        └── streamlit_interface.png
```

> **Note :** Le fichier modèle `EfficientNetB3_best.keras` (~43 Mo) n'est pas inclus dans le dépôt GitHub en raison de sa taille. Il doit être placé dans `2_Application_TriRetina/` avant de lancer l'application.

## Lancer l'application

```bash
cd TriRetina_GitHub/2_Application_TriRetina
pip install -r requirements.txt
streamlit run app.py
```

## Avertissement

Ce projet a été développé dans le cadre d'un stage en intelligence artificielle appliquée à la télé-ophtalmologie. Les résultats présentés sont expérimentaux et **ne constituent pas un diagnostic médical**.

## Auteur

**Kahina Ikerrouyene**
Stage en Intelligence Artificielle & Data — Télé-ophtalmologie, 2026

````

**Important :** le changement principal est bien celui-ci :

```markdown
<img src="TriRetina_GitHub/2_Application_TriRetina/streamlit_interface.png" alt="Interface web TriRetina" width="900">
````

Donc **supprime l'ancien chemin `assets/triRetina_interface.png`**. Ton image actuelle est directement dans `2_Application_TriRetina`, donc ce nouveau chemin correspond à ton dépôt réel.
