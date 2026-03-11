# SetVAE-Dental3D

Modèle SetVAE hiérarchique pour nuages de points dentaires 3D, avec prior en mélange de gaussiennes, inférence bidirectionnelle et visualisations d’attention multi‑niveaux permettant une génération coarse‑to‑fine et l’analyse de la structure latente des dents.

## Aperçu du projet

![Segmentation multi‑niveaux](./segmentation_multiniveau_cote_a_cote.png)

Cette image montre la **segmentation implicite multi‑niveaux** produite par le modèle.  
Chaque colonne correspond à un niveau latent du SetVAE (global → intermédiaire → fin).  
Les couleurs représentent l’**inducing point dominant** pour chaque point du nuage.

Cela permet d’observer comment le modèle décompose la dent en régions structurelles à différentes granularités.

---

## Description

Ce projet implémente une version hiérarchique du **SetVAE (CVPR 2021)** appliquée à des nuages de points dentaires 3D.  
Il inclut :

- un **prior hiérarchique** avec MoG au niveau supérieur,
- une **inférence bidirectionnelle** (top‑down + bottom‑up),
- un **décodeur coarse‑to‑fine** basé sur des blocs ABL,
- un **encodeur ISAB** pour traiter les sets,
- des **visualisations d’attention multi‑niveaux** (global → intermédiaire → fin),
- la génération de dents synthétiques.

Un fichier **tutoriel.pdf** est fourni pour expliquer les équations, la théorie du SetVAE, et comment l’implémentation correspond exactement au modèle mathématique.

---

## Installation

Cloner le dépôt :

```bash
git clone https://github.com/Yiphta-lab/SetVAE-Dental3D.git
cd <ton_repo>
```
Installer les dépendances :
```bash

pip install -r requirements.txt
```
Exécution

Le script principal est generation.py.

Pour lancer une génération :
```bash

python3 generation.py
```
Selon les options définies dans le script, cela :

 -   charge le modèle,

 -  génère un nuage de points,

 -   calcule les cartes d’attention,

 -   affiche les visualisations multi‑niveaux.
   
## Structure du dépôt

```text
.
├── generation.py        # Script principal
├── requirements.txt     # Dépendances
├── tutoriel.pdf         # Explication mathématique complète
├── image.png            # Image affichée dans le README
└── README.md
```
Dataset

Le modèle utilise le dataset 3DTeethSeg MICCAI, contenant des nuages de points dentaires segmentés.
Les chemins d’accès doivent être configurés dans generation.py.
Références

   - SetVAE: Learning Hierarchical Composition for Generative Modeling of Set-Structured Data, CVPR 2021.
     
   - 3DTeethSeg MICCAI Dataset.
