# TriRetina

Application de télé-ophtalmologie basée sur le deep learning pour la détection automatique de pathologies rétiniennes à partir de photos du fond d'œil.

## Pathologies détectées

- **Rétinopathie Diabétique (RD)**
- **Glaucome**
- **Dégénérescence Maculaire Liée à l'Âge (DMLA)**

## Architecture

- **Modèle** : EfficientNetB3 pré-entraîné (fine-tuning)
- **Dataset** : ~10 000 images de fond d'œil
- **Framework** : TensorFlow / Keras
- **Interface** : Streamlit

## Structure du projet

```
TriRetina/
├── 1_Notebook_Entrainement/   # Entraînement du modèle (Jupyter Notebook)
└── 2_Application_TriRetina/   # Application Streamlit
    ├── app.py
    ├── requirements.txt
    └── assets/
```

> **Note** : Le fichier modèle (`EfficientNetB3_best.keras`, ~43 Mo) n'est pas inclus dans ce dépôt en raison de sa taille. Il doit être placé dans `2_Application_TriRetina/` avant de lancer l'application.

## Lancer l'application

```bash
cd 2_Application_TriRetina
pip install -r requirements.txt
streamlit run app.py
```

## Auteur

Kahina Ikerrouyene — Stage en télé-ophtalmologie, 2026
