# Adversarially Robust Intrusion Detection System

**Cours** : ICCN - INE2 | **Projet 1**

## Description

Système de détection d'intrusion (IDS) basé sur le deep learning, conçu pour résister aux attaques adversariales. Un attaquant peut modifier légèrement les features du trafic réseau pour tromper un IDS classique — ce projet explore ces vulnérabilités et implémente des défenses robustes.

## Dataset

[UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) — trafic réseau labellisé (normal vs attaque), 49 features.

> Télécharger `UNSW_NB15_training-set.csv` et `UNSW_NB15_testing-set.csv` dans `data/`

## Structure

```
adversarial-IDS/
├── data/                     ← datasets CSV (non versionnés)
├── notebooks/
│   ├── 1_exploration.ipynb   ← exploration des données
│   ├── 2_baseline.ipynb      ← modèle MLP baseline
│   ├── 3_attacks.ipynb       ← attaques FGSM + PGD
│   ├── 4_defense.ipynb       ← adversarial training + feature squeezing
│   └── 5_results.ipynb       ← comparaison finale et graphiques
├── src/
│   ├── preprocessing.py      ← nettoyage et normalisation
│   ├── model.py              ← architecture MLP (PyTorch)
│   ├── attacks.py            ← FGSM, PGD
│   └── defense.py            ← défenses
├── results/                  ← graphiques et modèles sauvegardés
├── PROJECT1.md               ← plan de travail complet
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Lancer les notebooks

```bash
jupyter notebook
```

Ouvrir les notebooks dans l'ordre : `1_exploration` → `2_baseline` → `3_attacks` → `4_defense` → `5_results`

## Résultats

| Modèle | F1 (données propres) | F1 (sous FGSM) | F1 (sous PGD) |
|---|---|---|---|
| Baseline MLP | - | - | - |
| + Adversarial Training | - | - | - |
| + Feature Squeezing | - | - | - |

> Tableau mis à jour au fur et à mesure des expériences.

## Technologies

- Python 3.10+
- PyTorch
- scikit-learn
- pandas / numpy
- matplotlib / seaborn
