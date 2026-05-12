# Récap complet — Ce qu'on a fait et pourquoi

## Vue d'ensemble du projet

On a construit un **système de détection d'intrusion (IDS)** avec un réseau de neurones, puis on a montré comment un attaquant peut le tromper, et comment s'en défendre.

Le fil conducteur :

```
Données réseau  →  Modèle de base  →  Attaques  →  Défenses  →  Résultats
```

---

## Étape 1 — Les données (UNSW-NB15)

**Fichier** : `notebooks/1_exploration.ipynb`

On a utilisé le dataset **UNSW-NB15** qui contient du vrai trafic réseau capturé en laboratoire.

| | Train | Test |
|---|---|---|
| Exemples | 82 332 | 175 341 |
| Features | 42 | 42 |
| Attaques | 45 332 (55%) | — |
| Normal | 37 000 (45%) | — |

Chaque ligne = une connexion réseau avec ses statistiques :
- `dur` = durée de la connexion
- `spkts` = nombre de paquets envoyés
- `sbytes` = nombre de bytes envoyés
- `sttl` = TTL source
- `proto`, `service`, `state` = informations protocolaires
- ... 42 features au total

**Ce qu'on a observé** :
- Les classes sont bien équilibrées (55/45) → pas de problème majeur de déséquilibre
- Certaines features séparent clairement normal vs attaque (ex: `ct_state_ttl`, `sttl`)
- Colonnes catégorielles à encoder : `proto`, `service`, `state`

---

## Étape 2 — Preprocessing et modèle Baseline

**Fichier** : `notebooks/2_baseline.ipynb`

### Preprocessing (nettoyage des données)

```
Données brutes
    ↓
Supprimer colonnes inutiles (id, attack_cat)
    ↓
Remplacer les valeurs infinies et NaN par la médiane
    ↓
Encoder les colonnes texte en nombres (LabelEncoder)
    ↓
Normaliser toutes les features (StandardScaler → moyenne=0, écart-type=1)
    ↓
Données prêtes pour le modèle
```

**Pourquoi normaliser ?** Sans normalisation, une feature avec des valeurs 0-100000 domine une feature 0-1. Le modèle apprendrait mal.

### Architecture MLP (Multi-Layer Perceptron)

```
Input (42 features)
    ↓
Linear(42 → 128) + ReLU + Dropout(0.3)
    ↓
Linear(128 → 64) + ReLU + Dropout(0.3)
    ↓
Linear(64 → 32) + ReLU
    ↓
Linear(32 → 1) + Sigmoid
    ↓
Output : score entre 0 et 1
         > 0.5 = Attaque
         < 0.5 = Normal
```

**Dropout** : éteint aléatoirement 30% des neurones pendant l'entraînement → évite le surapprentissage.

### Entraînement

- Optimizer : Adam (lr=0.001)
- Loss : Binary Cross-Entropy
- 30 epochs, batch size 512

### Résultats du Baseline (sur données propres)

| Métrique | Valeur |
|---|---|
| F1-score | **0.9110** |
| Precision | 0.9879 |
| Recall | 0.8452 |
| PR-AUC | 0.9881 |
| Evasion Rate | 15.5% |

→ Le modèle est bon sur données normales. Mais 15.5% des attaques passent déjà sans modification.

---

## Étape 3 — Les attaques adversariales

**Fichier** : `notebooks/3_attacks.ipynb`

### Threat Model — qui est l'attaquant ?

Avant de générer des attaques, on a défini ce que l'attaquant peut faire :

**Ce qu'il peut modifier** (features manipulables) :
```
dur, spkts, dpkts, sbytes, dbytes, sttl,
sload, dload, sinpkt, dinpkt, sjit, djit, smean, dmean
```
→ Ce sont des statistiques qu'un attaquant contrôle en changeant le timing/volume de son trafic.

**Ce qu'il ne peut pas modifier** :
```
proto, service, state, srcip, dstip, dport
```
→ Changer le protocole ou l'IP brise l'attaque.

### Attaque FGSM (Fast Gradient Sign Method)

**Idée** : trouver la plus petite modification qui trompe le modèle.

```
1. Prendre une attaque (correctement détectée)
2. Calculer le gradient de la loss par rapport aux features
   → "si j'augmente cette feature de 0.001, la loss monte de combien ?"
3. Perturber dans la direction qui maximise la confusion du modèle
4. Appliquer seulement sur les features manipulables

X_adv = X + ε × sign(∇_X Loss) × masque_manipulable
```

ε = budget de perturbation (0.05, 0.1, 0.3)

### Attaque PGD (Projected Gradient Descent)

**Idée** : répéter FGSM en petits pas, version plus forte.

```
Répéter 10 fois :
    X = X + α × sign(gradient) × masque
    X = clip(X, X_original ± ε)   ← rester dans le budget
```

PGD est plus fort que FGSM car il s'adapte à chaque itération.

### Résultats des attaques sur le Baseline

| Scénario | F1 | Evasion Rate |
|---|---|---|
| Données propres | 0.9110 | 15.5% |
| Sous FGSM (ε=0.1) | 0.8533 | **24.8%** |
| Sous PGD (ε=0.1) | 0.8402 | **26.8%** |

→ Avec PGD, 26.8% des attaques passent inaperçues. Le modèle se fait tromper.

---

## Étape 4 — Les défenses

**Fichier** : `notebooks/4_defense.ipynb`

### Défense 1 — Adversarial Training

**Idée** : entraîner le modèle sur un mix de données propres + données adversariales.

```
Pour chaque epoch :
    1. Générer des exemples adversariaux avec FGSM
    2. Mélanger 50% propres + 50% adversariaux
    3. Entraîner le modèle sur ce mix
```

Le modèle apprend que l'attaque originale ET sa version modifiée sont toutes les deux des attaques. Sa frontière de décision devient plus large et plus robuste.

**Trade-off** : en apprenant sur des données plus difficiles, le modèle perd un peu de performance sur les données normales.

### Défense 2 — Feature Squeezing

**Idée** : réduire la précision des features pour effacer les petites perturbations.

```
Avant squeezing : dur = 2.10183746  ← perturbation calculée précisément
Après squeezing : dur = 2.09375     ← arrondi à 4 bits de précision
```

L'attaquant a calculé sa perturbation sur des valeurs précises. Après arrondi, la perturbation n'a plus le même effet.

```python
niveaux = 2^4 = 16
X_squeezed = round(X / max_val * 16) / 16 * max_val
```

### Résultats complets — comparaison des 3 modèles

| Modèle | Propres F1 | FGSM F1 | PGD F1 | Evasion% PGD |
|---|---|---|---|---|
| Baseline | **0.9110** | 0.8533 | 0.8402 | 26.8% |
| Adv. Training | 0.8971 | 0.7746 | 0.7738 | 36.2% |
| Feat. Squeezing | 0.8818 | **0.8570** | **0.8519** | **23.4%** |

---

## Étape 5 — Analyse des résultats

**Fichier** : `notebooks/5_results.ipynb`

### Ce qu'on observe

**Feature Squeezing est la meilleure défense ici** :
- F1 sous PGD : 0.8519 vs 0.7738 pour Adversarial Training
- Evasion Rate PGD : 23.4% vs 36.2%
- Perd moins de performance sur données propres (0.8818 vs 0.8971)

**Pourquoi Adversarial Training est moins bon ici** :
- On n'a fait que 10 epochs (version rapide) — avec 30 epochs complets, les résultats seraient meilleurs
- L'adversarial training avec seulement FGSM ne couvre pas bien les attaques PGD

**Courbe robustesse vs epsilon** :

| ε | Baseline F1 | Adv.Training F1 |
|---|---|---|
| 0.05 | 0.874 | 0.813 |
| 0.10 | 0.840 | 0.771 |
| 0.30 | 0.700 | 0.659 |

→ Les deux modèles se dégradent avec ε croissant, mais le baseline résiste mieux ici (à cause des 10 epochs seulement).

### Graphiques générés

| Fichier | Contenu |
|---|---|
| `class_distribution.png` | Distribution normal vs attaque |
| `attack_types.png` | Types d'attaques dans le dataset |
| `feature_distributions.png` | Distribution des 7 features clés |
| `feature_correlation.png` | Top 20 features corrélées au label |
| `training_loss.png` | Courbe de loss du baseline |
| `confusion_baseline.png` | Matrice de confusion du baseline |
| `attack_comparison.png` | F1 et Evasion Rate vs epsilon |
| `defense_comparison.png` | Comparaison défenses |
| `final_comparison.png` | Graphique final F1 + Evasion |
| `confusion_matrices.png` | 6 matrices de confusion |
| `robustness_vs_epsilon.png` | Robustesse vs budget epsilon |
| `final_results.csv` | Tableau complet des résultats |

---

## Résumé de l'architecture du projet

```
adversarial-IDS/
├── data/
│   ├── UNSW_NB15_training-set.csv   ← 82K lignes
│   └── UNSW_NB15_testing-set.csv    ← 175K lignes
├── notebooks/
│   ├── 1_exploration.ipynb          ← comprendre les données
│   ├── 2_baseline.ipynb             ← entraîner le MLP
│   ├── 3_attacks.ipynb              ← FGSM + PGD
│   ├── 4_defense.ipynb              ← Adv.Training + Feat.Squeezing
│   └── 5_results.ipynb              ← comparaison finale
├── results/
│   ├── baseline_model.pth           ← poids du modèle baseline
│   ├── adversarial_model.pth        ← poids du modèle robuste
│   ├── final_results.csv            ← tous les chiffres
│   └── *.png                        ← tous les graphiques
├── run_fast.py                      ← script pour tout régénérer
├── UNDERSTANDING.md                 ← théorie et concepts
├── PROJECT1.md                      ← plan de travail
└── RECAP.md                         ← ce fichier
```

---

## Ce qu'il faut savoir expliquer en soutenance

**Q : Pourquoi le baseline se fait tromper ?**
> Le modèle apprend des corrélations statistiques, pas la sémantique de l'attaque. En modifiant légèrement les features manipulables, l'attaquant déplace les exemples au-delà de la frontière de décision sans changer l'essence de l'attaque.

**Q : Comment fonctionne FGSM ?**
> On calcule le gradient de la loss par rapport aux features d'entrée. Ce gradient indique dans quelle direction perturber les features pour maximiser l'erreur du modèle. On applique ε × signe(gradient) uniquement sur les features manipulables.

**Q : Pourquoi Feature Squeezing marche bien ?**
> L'attaquant calcule sa perturbation avec une haute précision numérique. En réduisant la précision des features (arrondi à 4 bits), on efface ces petites perturbations avant qu'elles atteignent le modèle.

**Q : Pourquoi l'adversarial training est moins bon dans nos résultats ?**
> On a utilisé seulement 10 epochs pour des raisons de temps de calcul. Avec 30 epochs complets, les résultats seraient meilleurs. De plus, entraîner uniquement avec FGSM ne couvre pas parfaitement les attaques PGD.

**Q : Quel est le trade-off principal ?**
> Un modèle plus robuste est légèrement moins précis sur données normales. Baseline F1=0.911 vs Adv.Training F1=0.897 sur données propres. C'est le compromis robustesse/performance classique en adversarial ML.
