# TriRetina — Détection de Pathologies Rétiniennes

Application de télé-ophtalmologie par deep learning pour la détection automatique de trois pathologies à partir de photos du fond d'œil (rétinographie).

## Résultats obtenus

| Pathologie | AUC | Seuil optimal |
|---|---|---|
| Rétinopathie Diabétique | **0.9186** | 40 % |
| Glaucome | **0.9266** | 35 % |
| DMLA | **0.9175** | 45 % |
| **Moyenne** | **0.9209** | — |

## Modèle

- **EfficientNetB3** pré-entraîné sur ImageNet (fine-tuning bout-en-bout)
- Classification binaire indépendante par pathologie (sigmoid)
- Seuils optimisés sur la courbe ROC (maximisation du Youden Index)
- Score affiché normalisé : seuil → 50 % (au-dessus = signal d'alerte)

## Technologies

Python · TensorFlow / Keras · Streamlit · OpenCV · NumPy · fpdf2

## Dataset

~10 000 images de fond d'œil annotées (rétinographies couleur)  
Sources : APTOS 2019, REFUGE, ORIGA, DRISHTI, Kaggle EyePACS

## Structure du projet

```
TriRetina/
├── 1_Notebook_Entrainement/   # Entraînement EfficientNetB3 (Jupyter)
└── 2_Application_TriRetina/   # Interface Streamlit
    ├── app.py
    ├── requirements.txt
    └── assets/
```

> **Note** : Le fichier modèle (`EfficientNetB3_best.keras`, ~43 Mo) n'est pas inclus en raison de la limite GitHub. À placer dans `2_Application_TriRetina/` avant de lancer l'application.

## Lancer l'application

```bash
cd 2_Application_TriRetina
pip install -r requirements.txt
streamlit run app.py
```

## Auteur

Kahina Ikerrouyene — Stage en télé-ophtalmologie, 2026
