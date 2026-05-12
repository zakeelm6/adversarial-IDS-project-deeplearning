# Projet 1 — Adversarially Robust Intrusion Detection
**Cours** : ICCN - INE2 | **Binôme** : Arthur + partenaire

---

## C'est quoi le projet ?

On construit un **système de détection d'intrusion (IDS)** avec du deep learning.
Un IDS regarde le trafic réseau et dit : "c'est normal" ou "c'est une attaque".

Le problème : un attaquant peut **modifier légèrement** son trafic pour tromper l'IDS.
Notre objectif : rendre l'IDS **robuste** contre ces manipulations.

```
Trafic réseau  →  [Notre modèle IDS]  →  Normal / Attaque
                         ↑
              L'attaquant essaie de le tromper
              en modifiant les features du trafic
```

---

## Dataset choisi — UNSW-NB15

> Plus simple à télécharger et bien documenté. On part avec ça.

- Lien : https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Fichiers à télécharger : `UNSW_NB15_training-set.csv` + `UNSW_NB15_testing-set.csv`
- Les mettre dans `data/`
- ~49 features réseau (durée connexion, taille paquets, protocole, etc.)
- Label : `0` = normal, `1` = attaque

---

## Structure du repo

```
adversarial-IDS/
│
├── data/                        ← datasets CSV (ne pas push sur GitHub)
│   ├── UNSW_NB15_training-set.csv
│   └── UNSW_NB15_testing-set.csv
│
├── notebooks/
│   ├── 1_exploration.ipynb      ← Phase 1 : comprendre les données
│   ├── 2_baseline.ipynb         ← Phase 2 : entraîner le modèle de base
│   ├── 3_attacks.ipynb          ← Phase 3 : générer les attaques
│   ├── 4_defense.ipynb          ← Phase 4 : implémenter les défenses
│   └── 5_results.ipynb          ← Phase 5 : comparaison finale
│
├── src/
│   ├── preprocessing.py         ← fonctions de nettoyage des données
│   ├── model.py                 ← architecture du modèle MLP
│   ├── attacks.py               ← FGSM et PGD
│   └── defense.py               ← adversarial training + feature squeezing
│
├── results/                     ← graphiques et tableaux sauvegardés
│
├── requirements.txt
├── README.md
└── PROJECT1.md                  ← ce fichier
```

---

## Répartition du travail

| Tâche | Qui |
|---|---|
| Setup repo GitHub + structure | Les deux |
| Phase 1 : exploration des données | Arthur |
| Phase 2 : preprocessing + baseline | Les deux |
| Phase 3 : attaques FGSM/PGD | Binôme |
| Phase 4 : défenses | Arthur |
| Phase 5 : résultats + graphiques | Les deux |
| Rapport PDF | Les deux |

> Modifier cette répartition selon votre accord.

---

## Plan de travail — Phase par Phase

---

### PHASE 1 — Setup + Exploration des données
**But** : avoir l'environnement prêt et comprendre le dataset

#### Setup (faire une seule fois ensemble)

```bash
# 1. Installer les dépendances
pip install pandas numpy scikit-learn matplotlib seaborn jupyter torch torchvision

# 2. Créer le repo
mkdir adversarial-IDS && cd adversarial-IDS
mkdir data notebooks src results
git init

# 3. Créer .gitignore pour ne pas push les données
echo "data/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
```

#### Notebook `1_exploration.ipynb` — à coder

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
df = pd.read_csv('../data/UNSW_NB15_training-set.csv')

# 1. Taille du dataset
print("Shape:", df.shape)          # combien de lignes et colonnes

# 2. Voir les premières lignes
df.head()

# 3. Distribution des labels (normal vs attaque)
print(df['label'].value_counts())
sns.countplot(x='label', data=df)
plt.title('Normal (0) vs Attaque (1)')
plt.savefig('../results/class_distribution.png')
plt.show()

# 4. Valeurs manquantes
print(df.isnull().sum().sort_values(ascending=False).head(10))

# 5. Statistiques des features numériques
df.describe()
```

**Ce qu'on doit comprendre après cette phase :**
- Combien il y a d'attaques vs trafic normal (déséquilibre ?)
- Quelles features existent
- Y a-t-il des valeurs manquantes ou aberrantes

**Checklist Phase 1 :**
- [ ] Repo GitHub créé et partagé entre les deux
- [ ] Dataset téléchargé dans `data/`
- [ ] `1_exploration.ipynb` complété
- [ ] Graphique `class_distribution.png` sauvegardé

---

### PHASE 2 — Preprocessing + Modèle Baseline
**But** : nettoyer les données et entraîner un premier modèle qui fonctionne

#### Preprocessing — `src/preprocessing.py`

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

def load_and_clean(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    # Supprimer les colonnes inutiles
    drop_cols = ['id', 'attack_cat']  # à ajuster selon le dataset
    df_train.drop(columns=[c for c in drop_cols if c in df_train.columns], inplace=True)
    df_test.drop(columns=[c for c in drop_cols  if c in df_test.columns],  inplace=True)

    # Remplacer les infinis et NaN
    df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test.replace([np.inf, -np.inf],  np.nan, inplace=True)
    df_train.fillna(df_train.median(numeric_only=True), inplace=True)
    df_test.fillna(df_train.median(numeric_only=True), inplace=True)

    # Encoder les colonnes catégorielles (ex: proto, service, state)
    cat_cols = df_train.select_dtypes(include='object').columns.tolist()
    cat_cols = [c for c in cat_cols if c != 'label']
    le = LabelEncoder()
    for col in cat_cols:
        df_train[col] = le.fit_transform(df_train[col].astype(str))
        df_test[col]  = le.transform(df_test[col].astype(str).map(
            lambda x: x if x in le.classes_ else le.classes_[0]))

    # Séparer features et labels
    X_train = df_train.drop('label', axis=1).values.astype(np.float32)
    y_train = df_train['label'].values
    X_test  = df_test.drop('label', axis=1).values.astype(np.float32)
    y_test  = df_test['label'].values

    # Normalisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, scaler
```

#### Modèle MLP — `src/model.py`

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
```

#### Entraînement — `notebooks/2_baseline.ipynb`

```python
import torch
import numpy as np
import random
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

# Fixer les seeds — OBLIGATOIRE
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

import sys
sys.path.append('../src')
from preprocessing import load_and_clean
from model import MLP

# Charger les données
X_train, y_train, X_test, y_test, scaler = load_and_clean(
    '../data/UNSW_NB15_training-set.csv',
    '../data/UNSW_NB15_testing-set.csv'
)

# Convertir en tenseurs PyTorch
X_train_t = torch.FloatTensor(X_train)
y_train_t  = torch.FloatTensor(y_train)
X_test_t   = torch.FloatTensor(X_test)

# Créer le modèle
model = MLP(input_dim=X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Entraînement
for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    preds = model(X_train_t).squeeze()
    loss  = criterion(preds, y_train_t)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# Évaluation
model.eval()
with torch.no_grad():
    y_pred_prob = model(X_test_t).squeeze().numpy()
    y_pred = (y_pred_prob >= 0.5).astype(int)

print("F1    :", f1_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("PR-AUC   :", average_precision_score(y_test, y_pred_prob))

# Sauvegarder le modèle
torch.save(model.state_dict(), '../results/baseline_model.pth')
```

**Checklist Phase 2 :**
- [ ] `src/preprocessing.py` écrit et testé
- [ ] `src/model.py` écrit
- [ ] `2_baseline.ipynb` complété
- [ ] Résultats baseline notés ci-dessous :

| Métrique | Valeur |
|---|---|
| F1-score | ... |
| Precision | ... |
| Recall | ... |
| PR-AUC | ... |

---

### PHASE 3 — Attaques Adversariales
**But** : tromper le modèle baseline en modifiant les features de façon réaliste

#### Threat Model — à écrire dans le rapport

**Qui est l'attaquant ?**
- Accès : **White-box** (connaît le modèle) pour FGSM/PGD
- Objectif : faire classer une attaque comme "normal"
- Contrainte : les modifications doivent rester physiquement réalistes

**Features manipulables (UNSW-NB15) :**

| Feature | Manipulable ? | Raison |
|---|---|---|
| `dur` (durée connexion) | Oui | Facile à contrôler |
| `spkts` (nb paquets src) | Oui | Configurable |
| `sbytes` (nb bytes src) | Oui | Padding possible |
| `sttl` (TTL source) | Oui | Configurable |
| `sload` / `dload` | Oui | Débit configurable |
| `srcip` (IP source) | Non | Brise l'attaque |
| `dstip` (IP dest) | Non | Brise l'attaque |
| `proto` (protocole) | Non | Changer de protocole = nouvelle attaque |

#### Attaques — `src/attacks.py`

```python
import torch
import numpy as np

# Index des features manipulables dans le dataset (à ajuster)
MANIPULABLE_FEATURES = [0, 2, 4, 5, 7, 8]  # indices après preprocessing

def fgsm_attack(model, X, y, epsilon=0.1):
    """
    FGSM : Fast Gradient Sign Method
    Perturbe les features dans la direction qui maximise la loss
    """
    X_tensor = torch.FloatTensor(X).requires_grad_(True)
    y_tensor = torch.FloatTensor(y)

    output = model(X_tensor).squeeze()
    loss = torch.nn.BCELoss()(output, y_tensor)
    loss.backward()

    # Perturbation uniquement sur features manipulables
    perturbation = epsilon * X_tensor.grad.sign()
    mask = torch.zeros_like(perturbation)
    mask[:, MANIPULABLE_FEATURES] = 1.0
    perturbation = perturbation * mask

    X_adv = (X_tensor + perturbation).detach().numpy()
    return X_adv


def pgd_attack(model, X, y, epsilon=0.1, alpha=0.01, num_steps=40):
    """
    PGD : Projected Gradient Descent
    Version itérative de FGSM, plus puissante
    """
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    X_adv = X_tensor.clone()

    for _ in range(num_steps):
        X_adv = X_adv.detach().requires_grad_(True)
        output = model(X_adv).squeeze()
        loss = torch.nn.BCELoss()(output, y_tensor)
        loss.backward()

        # Pas de gradient sur features manipulables uniquement
        grad = X_adv.grad.sign()
        mask = torch.zeros_like(grad)
        mask[:, MANIPULABLE_FEATURES] = 1.0
        grad = grad * mask

        X_adv = X_adv + alpha * grad

        # Projection dans la boule epsilon
        delta = torch.clamp(X_adv - X_tensor, -epsilon, epsilon)
        X_adv = (X_tensor + delta).detach()

    return X_adv.numpy()
```

#### Test des attaques — `notebooks/3_attacks.ipynb`

```python
from attacks import fgsm_attack, pgd_attack

model.eval()

# Générer les exemples adversariaux (sur le test set)
X_fgsm = fgsm_attack(model, X_test, y_test, epsilon=0.1)
X_pgd  = pgd_attack(model,  X_test, y_test, epsilon=0.1)

# Évaluer sur les données adversariales
def evaluate(model, X, y_true):
    with torch.no_grad():
        preds_prob = model(torch.FloatTensor(X)).squeeze().numpy()
        preds = (preds_prob >= 0.5).astype(int)
    return {
        'F1':      f1_score(y_true, preds),
        'Evasion': 1 - f1_score(y_true[y_true==1], preds[y_true==1])
    }

print("Baseline  :", evaluate(model, X_test,  y_test))
print("Sous FGSM :", evaluate(model, X_fgsm,  y_test))
print("Sous PGD  :", evaluate(model, X_pgd,   y_test))
```

**Checklist Phase 3 :**
- [ ] Threat model rédigé (section dans le rapport)
- [ ] Tableau features manipulables complété
- [ ] `src/attacks.py` implémenté (FGSM + PGD)
- [ ] `3_attacks.ipynb` complété
- [ ] Résultats notés ci-dessous :

| Scénario | F1-score | Evasion Rate |
|---|---|---|
| Données propres | ... | - |
| Sous FGSM (ε=0.1) | ... | ...% |
| Sous PGD (ε=0.1) | ... | ...% |

---

### PHASE 4 — Défenses
**But** : rendre le modèle robuste aux attaques de la Phase 3

#### Défense 1 : Adversarial Training — `src/defense.py`

```python
def adversarial_training(model, X_train, y_train, epsilon=0.1, epochs=30):
    """
    Entraîne le modèle sur un mix de données propres + adversariales
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    for epoch in range(epochs):
        model.train()

        # Générer des exemples adversariaux pendant l'entraînement
        X_adv = fgsm_attack(model, X_train, y_train, epsilon=epsilon)

        # Mix 50/50 propres + adversariaux
        X_mixed = np.concatenate([X_train, X_adv])
        y_mixed = np.concatenate([y_train, y_train])

        # Shuffle
        idx = np.random.permutation(len(X_mixed))
        X_mixed, y_mixed = X_mixed[idx], y_mixed[idx]

        X_t = torch.FloatTensor(X_mixed)
        y_t = torch.FloatTensor(y_mixed)

        optimizer.zero_grad()
        preds = model(X_t).squeeze()
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    return model
```

#### Défense 2 : Feature Squeezing

```python
def feature_squeezing(X, n_bits=4):
    """
    Réduit la précision des features pour éliminer les petites perturbations
    n_bits=4 : garde seulement 16 niveaux de valeur par feature
    """
    max_val = np.max(np.abs(X), axis=0, keepdims=True) + 1e-8
    X_norm = X / max_val
    levels = 2 ** n_bits
    X_squeezed = np.round(X_norm * levels) / levels
    return X_squeezed * max_val
```

#### `notebooks/4_defense.ipynb`

```python
# Modèle avec adversarial training
model_robust = MLP(input_dim=X_train.shape[1])
model_robust = adversarial_training(model_robust, X_train, y_train, epsilon=0.1)
torch.save(model_robust.state_dict(), '../results/robust_model.pth')

# Évaluation complète
print("=== Modèle Robuste ===")
print("Données propres  :", evaluate(model_robust, X_test,  y_test))
print("Sous FGSM        :", evaluate(model_robust, X_fgsm,  y_test))
print("Sous PGD         :", evaluate(model_robust, X_pgd,   y_test))

# Avec Feature Squeezing
X_test_sq  = feature_squeezing(X_test)
X_fgsm_sq  = feature_squeezing(X_fgsm)
print("=== Feature Squeezing ===")
print("Données propres  :", evaluate(model, X_test_sq,  y_test))
print("Sous FGSM        :", evaluate(model, X_fgsm_sq,  y_test))
```

**Checklist Phase 4 :**
- [ ] `src/defense.py` implémenté
- [ ] `4_defense.ipynb` complété
- [ ] Résultats notés ci-dessous :

| Modèle | Données propres F1 | Sous FGSM F1 | Sous PGD F1 |
|---|---|---|---|
| Baseline (sans défense) | ... | ... | ... |
| + Adversarial Training | ... | ... | ... |
| + Feature Squeezing | ... | ... | ... |

---

### PHASE 5 — Résultats finaux + Rapport
**But** : produire les graphiques finaux et rédiger le rapport

#### `notebooks/5_results.ipynb` — graphiques à produire

```python
import matplotlib.pyplot as plt
import numpy as np

# Graphique 1 : comparaison des F1-scores
models    = ['Baseline', 'Adv. Training', 'Feature Squeezing']
f1_clean  = [0.95, 0.93, 0.94]   # remplacer avec vos valeurs réelles
f1_fgsm   = [0.40, 0.78, 0.65]
f1_pgd    = [0.30, 0.72, 0.60]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, f1_clean, width, label='Données propres', color='green')
ax.bar(x,         f1_fgsm,  width, label='Sous FGSM',       color='orange')
ax.bar(x + width, f1_pgd,   width, label='Sous PGD',        color='red')

ax.set_ylabel('F1-score')
ax.set_title('Robustesse des modèles sous attaque adversariale')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
plt.tight_layout()
plt.savefig('../results/robustness_comparison.png', dpi=150)
plt.show()
```

#### Structure du rapport (10 pages max)

| Section | Pages | Contenu |
|---|---|---|
| 1. Introduction | 0.5 | Problème, motivation, objectifs |
| 2. Threat Model | 1 | Qui est l'attaquant, features manipulables |
| 3. Dataset & Preprocessing | 1 | Description UNSW-NB15, étapes de nettoyage |
| 4. Modèle Baseline | 1.5 | Architecture MLP, résultats |
| 5. Attaques Adversariales | 2 | FGSM + PGD, résultats, analyse |
| 6. Défenses | 2 | Adversarial Training + Feature Squeezing, résultats |
| 7. Analyse de Robustesse | 1.5 | Tableaux comparatifs, graphiques, discussion |
| 8. Conclusion | 0.5 | Ce qui marche, limites, perspectives |

**Checklist Phase 5 :**
- [ ] `5_results.ipynb` avec tous les graphiques
- [ ] Rapport PDF rédigé (10 pages)
- [ ] README.md final avec instructions
- [ ] `requirements.txt` généré (`pip freeze > requirements.txt`)
- [ ] Repo GitHub propre et complet

---

## Variations expérimentales (obligatoires — minimum 3)

| # | Variation | Description |
|---|---|---|
| V1 | Budget d'attaque | Tester FGSM avec ε = 0.05, 0.1, 0.3 |
| V2 | Architecture | Comparer MLP 2 couches vs 4 couches |
| V3 | Défense | Comparer Adversarial Training vs Feature Squeezing |

---

## Reproductibilité — obligatoire

```python
# Toujours au début de chaque script/notebook
import random, numpy as np, torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
```

```bash
# requirements.txt — générer avec :
pip freeze > requirements.txt
```

---

## Points clés pour la soutenance

- Expliquer pourquoi certaines features sont manipulables et d'autres non
- Expliquer FGSM en une phrase : "on perturbe dans la direction qui augmente la loss"
- Expliquer le trade-off : l'adversarial training améliore la robustesse mais peut baisser légèrement les performances sur données propres
- Savoir lire un tableau de robustesse et commenter les résultats

---

## Avancement global

- [ ] Phase 1 — Setup + Exploration
- [ ] Phase 2 — Preprocessing + Baseline
- [ ] Phase 3 — Attaques (FGSM + PGD)
- [ ] Phase 4 — Défenses
- [ ] Phase 5 — Résultats + Rapport
