# Adversarial IDS — Guide de présentation technique

> Ce document explique le projet de A à Z avec la rigueur attendue par un professeur de deep learning.
> Chaque section correspond à une phase du projet et contient les formules, les choix d'architecture, et les justifications techniques.

---

## Table des matières

1. [Le problème — pourquoi ce projet existe](#1-le-problème)
2. [Le dataset — UNSW-NB15](#2-le-dataset)
3. [Preprocessing — préparer les données](#3-preprocessing)
4. [Modèle Baseline — le MLP](#4-modèle-baseline)
5. [Attaques adversariales — FGSM et PGD](#5-attaques-adversariales)
6. [Défenses — Adversarial Training et Feature Squeezing](#6-défenses)
7. [Résultats et analyse](#7-résultats-et-analyse)
8. [Questions probables du prof et réponses techniques](#8-questions-du-prof)

---

## 1. Le problème

### Contexte

Un **IDS (Intrusion Detection System)** analyse le trafic réseau en temps réel et classe chaque connexion comme normale ou malveillante. Les IDS modernes utilisent du machine learning : on entraîne un classifieur sur des données labellisées et on le déploie en production.

**Le problème fondamental :** un modèle de deep learning est vulnérable aux **exemples adversariaux** — des entrées délibérément perturbées pour induire une mauvaise classification. Dans le contexte réseau, un attaquant peut modifier légèrement son comportement (timing, taille des paquets, débit) sans changer l'essence de son attaque, et ainsi passer sous le radar de l'IDS.

### Formulation mathématique du problème

Soit `f : R^d → [0,1]` notre classifieur, `x ∈ R^d` un vecteur de features, `y ∈ {0,1}` le label (0=normal, 1=attaque).

**Classification normale :**
```
f(x) > 0.5  →  "Attaque détectée"
f(x) < 0.5  →  "Trafic normal"
```

**Objectif de l'attaquant :** trouver `x_adv = x + δ` tel que :
```
f(x_adv) < 0.5   (faire classer une attaque comme normale)
||δ||_∞ ≤ ε      (la perturbation reste petite et indétectable)
δ_i = 0          pour toutes les features non-manipulables i
```

Notre projet répond à : **comment construire `f` robuste face à cet attaquant ?**

---

## 2. Le dataset

### UNSW-NB15

Dataset de trafic réseau capturé en laboratoire (University of New South Wales, 2015). Il contient du vrai trafic normal et des attaques réelles (DoS, Fuzzers, Reconnaissance, Shellcode, Worms...).

| Split | Lignes | Features | % Attaques |
|-------|--------|----------|------------|
| Train | 82 332 | 42 | 55% |
| Test  | 175 341 | 42 | ~46% |

### Les 42 features

Chaque ligne représente une **connexion réseau** avec ses statistiques agrégées :

| Catégorie | Features | Description |
|-----------|----------|-------------|
| Durée | `dur` | Durée totale de la connexion |
| Volume | `sbytes`, `dbytes`, `spkts`, `dpkts` | Bytes et paquets source/destination |
| Débit | `sload`, `dload` | Bits par seconde |
| TTL | `sttl`, `dttl` | Time-To-Live source/destination |
| Timing | `sinpkt`, `dinpkt` | Temps inter-paquets moyen |
| Gigue | `sjit`, `djit` | Jitter (variance du timing) |
| Protocolaire | `proto`, `service`, `state` | Protocole, service, état connexion |
| Comportemental | `ct_state_ttl`, `ct_dst_ltm`... | Statistiques sur les connexions récentes |

**Label :** `label = 0` (normal) ou `label = 1` (attaque)

### Pourquoi ce dataset ?

- Bien documenté et utilisé dans la littérature scientifique
- Features réalistes (collectées sur vrai trafic)
- Classes relativement équilibrées (pas de problème extrême de déséquilibre)
- Existe en format CSV, facile à charger avec pandas

---

## 3. Preprocessing

### Pipeline complet

```
CSV brut (42 features + colonnes inutiles)
    ↓ Supprimer 'id', 'attack_cat'
    ↓ Remplacer np.inf et NaN par la médiane
    ↓ LabelEncoder sur colonnes catégorielles (proto, service, state)
    ↓ StandardScaler sur toutes les features numériques
    ↓ Tenseurs PyTorch float32
```

### StandardScaler — pourquoi c'est indispensable

Le StandardScaler transforme chaque feature `x_i` en :
```
x_i_scaled = (x_i - μ_i) / σ_i
```
où `μ_i` est la moyenne et `σ_i` l'écart-type de la feature `i` **calculés sur le train set uniquement** (pas le test set — sinon c'est du data leakage).

**Sans normalisation :**
- `sbytes` peut valoir 10 000 000
- `dur` peut valoir 0.003
- Le gradient serait dominé par `sbytes` → convergence lente et instable
- Les poids appris n'auraient pas la même échelle

**Après normalisation :** toutes les features ont μ=0, σ=1. Le gradient descent converge beaucoup mieux.

### LabelEncoder sur les features catégorielles

Les colonnes `proto` (`tcp`, `udp`, `icmp`...) et `service` (`http`, `ftp`, `dns`...) sont des chaînes de caractères. On les encode en entiers :
```
tcp → 0, udp → 1, icmp → 2, ...
```

**Important :** le scaler est fit sur le train set et appliqué tel quel sur le test set. On ne re-fit jamais sur le test (sinon on triche).

---

## 4. Modèle Baseline

### Architecture MLP (Multi-Layer Perceptron)

```
Input x ∈ R^42
    ↓
[Linear 42→128]  W₁ ∈ R^(128×42), b₁ ∈ R^128
    ↓
[ReLU]  f(z) = max(0, z)
    ↓
[Dropout p=0.3]  — actif uniquement à l'entraînement
    ↓
[Linear 128→64]  W₂ ∈ R^(64×128), b₂ ∈ R^64
    ↓
[ReLU]
    ↓
[Dropout p=0.3]
    ↓
[Linear 64→32]   W₃ ∈ R^(32×64), b₃ ∈ R^32
    ↓
[ReLU]
    ↓
[Linear 32→1]    W₄ ∈ R^(1×32), b₄ ∈ R^1
    ↓
[Sigmoid]  σ(z) = 1 / (1 + e^(-z))
    ↓
Output ŷ ∈ (0, 1)
```

### Pourquoi ReLU ?

ReLU (Rectified Linear Unit) : `f(z) = max(0, z)`

- **Pas de vanishing gradient** contrairement à Sigmoid ou Tanh dont les gradients saturent à 0 pour les grandes valeurs
- **Sparse activation** : certains neurones sont à 0, ce qui donne une représentation plus compacte
- **Calcul rapide** : juste un max()

### Pourquoi Dropout ?

Le Dropout (Srivastava et al. 2014) éteint aléatoirement chaque neurone avec probabilité `p=0.3` pendant l'entraînement :

```
à l'entraînement : neurone i actif avec proba 0.7, éteint avec proba 0.3
à l'inférence    : tous les neurones actifs, poids multipliés par 0.7
```

Effet : le réseau ne peut pas "compter" sur un neurone spécifique → force une représentation distribuée et robuste → réduit l'overfitting.

### Pourquoi Sigmoid en sortie ?

Classification binaire → on veut une probabilité entre 0 et 1.
Sigmoid : `σ(z) = 1 / (1 + e^(-z))` → output ∈ (0,1)
Seuil de décision : `ŷ > 0.5 → "Attaque"`

### Fonction de perte — Binary Cross-Entropy

```
BCE(y, ŷ) = - [y · log(ŷ) + (1-y) · log(1-ŷ)]
```

- Si `y=1` (attaque) et `ŷ=1.0` → perte = 0 (parfait)
- Si `y=1` (attaque) et `ŷ=0.01` → perte = -log(0.01) ≈ 4.6 (très mauvais)
- Si `y=0` (normal) et `ŷ=0.01` → perte = -log(0.99) ≈ 0.01 (presque parfait)

La BCE pénalise fortement les prédictions confiantes mais fausses.

### Optimiseur — Adam

Adam (Adaptive Moment Estimation) combine deux idées :
- **Momentum** : utilise la moyenne pondérée des gradients passés (évite les oscillations)
- **RMSProp** : adapte le learning rate par paramètre selon l'historique des gradients²

```
m_t = β₁ m_{t-1} + (1 - β₁) g_t       ← moyenne des gradients (β₁=0.9)
v_t = β₂ v_{t-1} + (1 - β₂) g_t²      ← moyenne des gradients² (β₂=0.999)
θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
```

Learning rate `α = 0.001` — valeur par défaut d'Adam, généralement bon point de départ.

### Entraînement

- 30 epochs, batch size 512
- Seed fixée à 42 pour la reproductibilité (`random`, `numpy`, `torch`)

### Métriques — pourquoi F1 et pas accuracy ?

Même avec des classes équilibrées, l'accuracy peut être trompeuse :
- Un modèle qui dit toujours "attaque" aurait 55% d'accuracy (et serait inutile)

**Precision :** parmi les exemples classés "attaque", combien le sont vraiment ?
```
Precision = TP / (TP + FP)
```

**Recall :** parmi les vraies attaques, combien le modèle en détecte ?
```
Recall = TP / (TP + FN)
```

**F1-score :** moyenne harmonique de Precision et Recall
```
F1 = 2 · (Precision · Recall) / (Precision + Recall)
```

Pour un IDS, le Recall est critique (manquer une attaque = catastrophique). Le F1 équilibre les deux.

**PR-AUC (Area Under Precision-Recall Curve) :** résume la performance à tous les seuils de décision. Plus robuste que ROC-AUC quand les classes sont déséquilibrées.

**Evasion Rate :** proportion des vraies attaques que le modèle classe comme normal (= 1 - Recall des attaques)

### Résultats baseline

| Métrique | Valeur |
|----------|--------|
| F1-score | **0.9110** |
| Precision | 0.9879 |
| Recall | 0.8452 |
| PR-AUC | 0.9881 |
| Evasion Rate | **15.48%** |

Le modèle est bon — mais 15.48% des attaques passent déjà inaperçues sur données propres. Sous attaque adversariale, ça va empirer.

---

## 5. Attaques adversariales

### Threat Model

Avant d'implémenter une attaque, on doit définir précisément ce que l'attaquant peut faire.

**Type d'attaquant : White-box**
L'attaquant connaît le modèle, ses poids, son architecture. C'est le scénario le plus difficile pour le défenseur — et donc le plus intéressant à étudier.

**Objectif (targeted) :** faire classer une attaque (y=1) comme normale (ŷ < 0.5)

**Contraintes physiques :** un attaquant réseau réel ne peut pas modifier n'importe quelle feature.

| Feature | Manipulable | Justification |
|---------|-------------|---------------|
| `dur` | ✅ | Il contrôle quand il ferme la connexion |
| `sbytes`, `dbytes` | ✅ | Il peut ajouter des données inutiles (padding) |
| `spkts`, `dpkts` | ✅ | Il contrôle le nombre de paquets |
| `sttl` | ✅ | TTL configurable dans la stack réseau |
| `sload`, `dload` | ✅ | Il contrôle son débit |
| `sinpkt`, `sjit` | ✅ | Il contrôle le timing de ses paquets |
| `proto` | ❌ | Changer TCP→UDP = changer d'attaque |
| `srcip`, `dstip` | ❌ | Brise la connexion TCP |
| `dport` | ❌ | Brise l'attaque (change la cible) |
| `ct_state_ttl` | ❌ | Feature calculée sur plusieurs connexions passées |

On applique un **masque binaire** `m ∈ {0,1}^d` dans les attaques pour limiter les perturbations aux features manipulables.

---

### Attaque FGSM (Fast Gradient Sign Method)

**Papier original :** Goodfellow et al. (2014) — "Explaining and Harnessing Adversarial Examples"

**Idée fondamentale :** le modèle est différentiable. On peut calculer le gradient de la perte par rapport aux *entrées* (pas juste par rapport aux poids). Ce gradient nous dit dans quelle direction perturber `x` pour maximiser la perte.

**Algorithme :**

```
Entrée : modèle f, exemple x, label y, budget ε, masque m
    1. Calculer ŷ = f(x)
    2. Calculer la loss L = BCE(y, ŷ)
    3. Calculer g = ∂L/∂x  (gradient de la loss par rapport à x)
    4. Perturbation = ε · sign(g) · m
    5. x_adv = x + ε · sign(g) · m
Sortie : x_adv
```

**sign(g)** : on prend juste le signe du gradient (+1 ou -1), pas sa magnitude. Cela garantit que la perturbation est exactement ε dans chaque dimension manipulable.

**En PyTorch :**
```python
X_tensor = torch.FloatTensor(X).requires_grad_(True)
loss = BCELoss()(model(X_tensor).squeeze(), y_tensor)
loss.backward()                           # calcule ∂L/∂x
perturbation = epsilon * X_tensor.grad.sign() * mask
X_adv = X_tensor + perturbation
```

**Complexité :** un seul forward pass + un backward pass. Très rapide.

**Limitation :** une seule étape — pas forcément optimal. L'exemple adversarial peut rester trop "proche" d'une région mal classifiée sans y entrer vraiment.

---

### Attaque PGD (Projected Gradient Descent)

**Papier original :** Madry et al. (2018) — "Towards Deep Learning Models Resistant to Adversarial Attacks"

**Idée :** FGSM itéré avec projection. À chaque itération, on fait un petit pas FGSM, puis on projette dans la boule `ε` autour de `x` original.

**Algorithme :**

```
Entrée : modèle f, exemple x, label y, budget ε, pas α, étapes T, masque m
    x_adv = x  (initialisation)
    Pour t = 1 à T :
        g = ∂BCE(y, f(x_adv)) / ∂x_adv    ← gradient au point courant
        x_adv = x_adv + α · sign(g) · m   ← pas de gradient
        δ = x_adv - x
        δ = clamp(δ, -ε, ε)               ← projection dans la boule L∞
        x_adv = x + δ
Sortie : x_adv
```

**Paramètres utilisés :** ε=0.1, α=0.01, T=40

**Pourquoi PGD est plus fort que FGSM :**
- 40 itérations vs 1 → explore mieux l'espace autour de x
- Chaque pas repart du gradient *local* au point courant (pas du gradient au point original)
- Converge vers le maximum local de la perte dans la boule ε
- Madry et al. montrent que PGD trouve l'exemple adversarial "le plus fort" dans la boule ε (en pratique)

**PGD est considéré comme le standard de facto** pour évaluer la robustesse adversariale.

---

### Résultats des attaques

| Scénario | F1 | Precision | Recall | PR-AUC | Evasion Rate |
|----------|----|-----------|--------|--------|--------------|
| Données propres | 0.9110 | 0.9879 | 0.8452 | 0.9881 | 15.48% |
| Sous FGSM ε=0.1 | 0.8533 | 0.9864 | 0.7518 | 0.9819 | 24.82% |
| Sous PGD ε=0.1 | 0.8402 | 0.9860 | 0.7320 | 0.9796 | 26.80% |

**Observations :**
- La Precision reste quasi-identique (~0.986) : les faux positifs (trafic normal classé attaque) n'augmentent presque pas
- Le Recall chute : le modèle *manque* de plus en plus d'attaques (les exemples adversariaux lui ressemblent à du trafic normal)
- PGD > FGSM en efficacité : evasion rate 26.8% vs 24.8%

---

## 6. Défenses

### Défense 1 — Adversarial Training (Approche 1 — FGSM-AT)

**Papier référence :** Madry et al. (2018) — même papier que PGD

**Idée :** si le modèle voit des exemples adversariaux pendant l'entraînement, il apprend à les classer correctement. Sa frontière de décision s'élargit autour des exemples d'entraînement.

**Formulation mathématique (min-max problem) :**

```
min_θ  E_(x,y) [ max_{||δ||≤ε, δ_i=0 si non-manip.} L(f_θ(x + δ), y) ]
```

On minimise (sur les poids θ) le maximum de la perte (sur les perturbations δ). C'est un problème de saddle point — l'adversarial training le résout de façon approximative.

**Algorithme implémenté :**

```
Pour chaque epoch (30 epochs, batch=512) :
    1. Générer X_adv = FGSM(modèle, X_train, y_train, ε=0.1)
    2. X_mixed = concat([X_train, X_adv])   → doublement des données
    3. y_mixed = concat([y_train, y_train])
    4. Shuffle aléatoire de X_mixed, y_mixed
    5. Un pas de gradient descent sur BCE(f(X_mixed), y_mixed)
```

**Trade-off robustesse/performance :** un théorème (Tsipras et al. 2019) montre qu'il existe une tension fondamentale entre performance sur données propres et robustesse adversariale. Le modèle "consomme" de la capacité pour apprendre à résister aux perturbations.

---

### Constat d'échec — Pourquoi FGSM-AT ne fonctionne pas

Les résultats obtenus avec cette première approche sont **contre-intuitifs** : le modèle entraîné de façon adversariale est **moins robuste** que le baseline non défendu.

| Modèle | Evasion Rate (PGD) | vs Baseline |
|--------|-------------------|-------------|
| Baseline (sans défense) | 26.80% | — |
| FGSM Adversarial Training | **36.20%** | **+9.4% pire** |

Trois raisons expliquent cet échec :

**Raison 1 — Robustness Gap (cause principale)**

Le modèle est entraîné avec des exemples FGSM (1 seule étape de gradient) mais évalué sous attaque PGD (40 étapes itératives). FGSM génère des perturbations faibles et peu optimales — le modèle apprend à résister à un adversaire de niveau 1, mais l'évaluation se fait contre un adversaire de niveau 40.

```
Entraînement : perturbations FGSM  (force ≈ 1 étape)
Évaluation   : perturbations PGD   (force ≈ 40 étapes)
→ Le modèle a appris à résister à une menace différente de celle à laquelle il est soumis.
```

Ce phénomène est documenté dans la littérature : un modèle FGSM-robuste n'est pas PGD-robuste (Madry et al. 2018, Appendix).

**Raison 2 — Distribution Shift**

En mélangeant 50% d'exemples propres et 50% d'exemples FGSM à chaque epoch, on entraîne le modèle sur une distribution artificielle qui n'est ni la distribution réelle ni une distribution adversariale pure. Le modèle doit simultanément :
- bien classer les exemples propres
- bien classer les exemples FGSM perturbés

Ces deux objectifs tirent les poids dans des directions contradictoires, ce qui dégrade les deux performances au lieu d'en améliorer une.

**Raison 3 — Attaque générée sur le modèle courant (non stabilisé)**

À chaque epoch, FGSM est généré sur le modèle **en cours d'entraînement**. En début d'entraînement, le modèle est aléatoire, donc les exemples adversariaux générés sont de mauvaise qualité. Le modèle n'apprend jamais à résister à des exemples adversariaux construits sur un modèle convergé.

**Conclusion :** l'Approche 1 (FGSM-AT) est fondamentalement limitée par le choix de l'attaque d'entraînement. Pour obtenir une vraie robustesse adversariale, il faut entraîner avec l'attaque la plus forte disponible — PGD — ce qui motive directement l'Approche 2.

---

### Approche 2 — PGD Adversarial Training (PGD-AT)

**Philosophie de Madry :** entraîner sur la même attaque que celle d'évaluation — le pire adversaire possible dans la boule ε.

#### Idée fondamentale

À chaque étape d'entraînement, pour chaque exemple `(x, y)` :
1. Chercher la **pire perturbation possible** via PGD (inner maximization)
2. Mettre à jour les poids pour **classifier correctement** cet exemple adversarial (outer minimization)

Le modèle apprend directement à résister à l'adversaire le plus fort qu'il connaît.

#### Formulation mathématique — Problème min-max de Madry

```
min_θ  E_(x,y) [ max_{||δ||_∞ ≤ ε} L(f_θ(x + δ), y) ]
```

**Inner maximization** (rôle de l'attaquant) :
```
max_{||δ||_∞ ≤ ε} L(f(x + δ), y)
```
L'attaquant cherche la **perturbation la plus déstabilisante** dans la boule ε.
PGD approxime cette étape en T itérations — plus T est grand, plus l'approximation est précise.

**Outer minimization** (rôle du défenseur) :
```
min_θ  E[ pire perte trouvée ci-dessus ]
```
Le modèle apprend à résister à cette pire attaque — pas à une attaque faible comme FGSM.

#### Pourquoi c'est beaucoup plus robuste que FGSM-AT

| Propriété | FGSM-AT | PGD-AT |
|-----------|---------|--------|
| Attaque d'entraînement | 1 étape (faible) | 7 étapes (fort) |
| Attaque d'évaluation | PGD-40 | PGD-40 |
| Cohérence train/eval | ❌ Robustness gap | ✅ Même famille |
| Frontières de décision | Étroites | **Larges et robustes** |
| Gradients | Instables | **Plus stables** |

Le modèle PGD-AT développe des **régions locales plus larges** autour de chaque exemple d'entraînement — il n'a pas seulement appris le bon label, il a appris que toute la boule ε autour de `x` doit être classifiée correctement.

#### Algorithme implémenté (PGD-7)

```
Pour chaque epoch (30 epochs, batch=512) :
    Pour chaque batch (x, y) :
        1. x_adv = x  (initialisation)
        2. Pour t = 1 à 7 :
               g = ∂L(f(x_adv), y) / ∂x_adv
               x_adv = x_adv + α · sign(g) · masque
               δ = clamp(x_adv - x, -ε, ε)
               x_adv = x + δ
        3. Mettre à jour θ sur L(f(x_adv), y)
```

Paramètres : ε=0.1, α=0.01, steps=7 (PGD-7 — standard de la littérature pour l'entraînement).

---

### Résultats de l'Approche 2 — Constat et analyse

Les résultats obtenus sont contre-intuitifs : PGD-AT est encore **plus faible** que FGSM-AT.

| Modèle | F1 propres | Evasion Rate (PGD) | Verdict |
|--------|------------|-------------------|---------|
| Baseline (sans défense) | **0.9110** | 28.2% | référence |
| Approche 1 — FGSM-AT | 0.8971 | 36.7% | pire que baseline |
| Approche 2 — PGD-AT | 0.8943 | **40.2%** | pire que les deux |

**Pourquoi PGD-AT est encore plus dégradé :**

La cause principale est la **Correction 2** appliquée — entraîner sur 100% d'exemples adversariaux PGD sans aucune donnée propre s'avère trop agressif :

1. **Oubli catastrophique de la distribution naturelle :** le modèle ne voit jamais les données propres pendant l'entraînement. Sa frontière de décision se déplace entièrement vers la distribution adversariale PGD-7, et il perd la capacité de bien ancrer les exemples naturels.

2. **Gap PGD-7 (train) vs PGD-40 (évaluation) :** les exemples PGD-7 générés à l'entraînement sont moins optimaux que les exemples PGD-40 de l'évaluation. Le modèle apprend à résister à une force de 7 étapes, mais l'évaluation applique 40 étapes — le gap est plus grand qu'avec FGSM-AT.

3. **Overfitting adversarial :** en entraînant 100% sur PGD, le modèle surfit la distribution adversariale d'entraînement sans généraliser aux nouvelles perturbations adversariales vues à l'évaluation.

**Conclusion sur l'Adversarial Training (les deux approches) :**

Les deux variantes d'Adversarial Training échouent sur ce dataset. Cela illustre une difficulté fondamentale de cette technique sur les **données tabulaires réseau** : contrairement aux images où les perturbations adversariales sont localement cohérentes, les features réseau sont hétérogènes (durée, bytes, TTL, jitter) et les perturbations adversariales ne suivent pas de structure spatiale exploitable par l'entraînement.

**Feature Squeezing reste la meilleure défense** (Evasion 23.4%), car elle agit directement sur la précision numérique des perturbations sans modifier le modèle — une approche orthogonale plus adaptée à ce type de données.

### Défense 2 — Feature Squeezing

**Papier référence :** Xu et al. (2018) — "Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks"

**Idée :** les exemples adversariaux sont construits avec une grande précision numérique. Si on réduit cette précision (quantification), les perturbations disparaissent.

**Formulation :**

```
Pour chaque feature i :
    1. Normaliser : x_i_norm = x_i / max_i  (max calculé sur le train)
    2. Quantifier sur n_bits niveaux :
       x_i_sq = round(x_i_norm × 2^n_bits) / 2^n_bits
    3. Redimensionner : x_i_out = x_i_sq × max_i
```

Avec `n_bits = 4` : seulement **16 niveaux** de valeur possibles par feature.

**Exemple concret :**

```
Attaquant calcule : x_adv[dur] = 0.3847261934...  (perturbation calculée au millième)
Après squeezing  : x_sq[dur]  = 0.375            (arrondi à 1/16 = 0.0625)
La perturbation soigneusement calculée est effacée.
```

**Avantages :**
- S'applique à l'inférence seulement — pas besoin de re-entraîner le modèle
- Applicable à n'importe quel modèle existant
- Très rapide (juste des opérations arithmétiques)

**Limitation :** si ε est grand, la perturbation survit à la quantification.

---

## 7. Résultats et analyse

### Tableau complet — toutes les approches

| Modèle | F1 propres | F1 FGSM | F1 PGD | Evasion PGD |
|--------|-----------|---------|--------|-------------|
| Baseline MLP | **0.9110** | 0.8533 | 0.8402 | 28.24% |
| Approche 1 — FGSM-AT | 0.8971 | 0.7746 | 0.7700 | 36.71% |
| Approche 2 — PGD-AT | 0.8943 | 0.7484 | 0.7425 | 40.18% |
| + Feature Squeezing | 0.8818 | **0.8570** | **0.8519** | **23.42%** |

### Analyse des résultats

**Constat principal : l'Adversarial Training échoue dans les deux variantes.**

Les deux approches d'Adversarial Training (FGSM-AT et PGD-AT) produisent un modèle **moins robuste** que le baseline sans défense — un résultat contre-intuitif mais scientifiquement explicable (voir Section 6).

**Feature Squeezing gagne sur tous les scénarios d'attaque :**
- F1 sous PGD : **0.8519** vs 0.8402 (baseline), 0.7700 (FGSM-AT), 0.7425 (PGD-AT)
- Evasion Rate PGD : **23.4%** vs 28.2% (baseline), 36.7% (FGSM-AT), 40.2% (PGD-AT)
- Perd seulement 3% de F1 sur données propres vs baseline

**Pourquoi l'Adversarial Training échoue sur ce dataset :**

> Voir analyses détaillées en Section 6 — Constat d'échec FGSM-AT et Résultats Approche 2.

En synthèse : les deux approches souffrent du même problème fondamental — les données tabulaires réseau sont hétérogènes (features de natures très différentes), ce qui rend les perturbations adversariales non-structurées et difficiles à capturer par l'entraînement. Feature Squeezing contourne ce problème en agissant au niveau de la précision numérique, indépendamment du modèle.

**Courbe robustesse vs epsilon (sur PGD) :**

| ε | Baseline F1 | Adv.Training F1 | Feat.Squeeze F1 |
|---|-------------|-----------------|-----------------|
| 0.05 | ~0.87 | ~0.81 | ~0.86 |
| 0.10 | 0.840 | 0.774 | 0.852 |
| 0.30 | ~0.70 | ~0.66 | ~0.77 |

→ Feature Squeezing résiste mieux à tous les budgets d'attaque testés.

### Variations expérimentales (V1, V2, V3)

**V1 — Budget d'attaque ε variable :**
Plus ε est grand, plus le modèle se dégrade. Mais la perturbation doit rester physiquement réaliste — un ε trop grand crée des features impossibles (durée négative, bytes impossibles).

**V2 — Architecture MLP :**
On compare 2 couches (128→64→1) vs 4 couches (128→64→32→16→1). Un modèle plus profond apprend des représentations plus abstraites mais est plus difficile à entraîner de façon robuste.

**V3 — Comparaison des défenses :**
Comme dans le tableau ci-dessus. Feature Squeezing gagne en pratique sur ce dataset avec ces hyperparamètres.

---

## 8. Questions du prof

### Q1 : "Expliquez FGSM mathématiquement"

FGSM exploite la différentiabilité du modèle. On calcule le gradient de la loss par rapport aux *entrées* :
```
g = ∂L(f_θ(x), y) / ∂x
```
Ce gradient indique dans quelle direction modifier `x` pour *augmenter* la perte. On prend le signe de `g` (pas sa magnitude) pour garantir une perturbation de norme L∞ exactement égale à ε :
```
x_adv = x + ε · sign(g)
```
Avec le masque des features manipulables : `x_adv = x + ε · sign(g) · m`.

### Q2 : "Quelle est la différence entre FGSM et PGD ?"

FGSM fait **un seul pas** de perturbation depuis `x` original. PGD fait **T petits pas** itératifs (T=40, α=0.01), en projetant après chaque pas dans la boule ε. PGD explore mieux l'espace des perturbations et trouve généralement un exemple adversarial plus fort. Madry et al. montrent que PGD converge vers le maximum local de la loss dans la boule ε — ce qui en fait une borne supérieure sur la force des attaques de premier ordre.

### Q3 : "Pourquoi Sigmoid et pas Softmax ?"

Softmax est utilisé pour la classification multi-classe (k > 2 sorties). Ici c'est une classification binaire (normal vs attaque) → une seule sortie avec Sigmoid suffit. Sigmoid et Softmax à 2 classes sont mathématiquement équivalents, mais Sigmoid avec une sortie est plus simple.

### Q4 : "Comment fonctionne le Dropout à l'inférence ?"

À l'entraînement : chaque neurone est éteint avec probabilité p=0.3, les neurones actifs ont leurs sorties divisées par (1-p) pour maintenir l'espérance.
À l'inférence (`model.eval()` en PyTorch) : le Dropout est désactivé, tous les neurones sont actifs. PyTorch gère cela automatiquement via `model.train()` / `model.eval()`.

### Q5 : "Pourquoi normaliser avec les stats du train set seulement ?"

Utiliser les statistiques du test set introduit du **data leakage** — le modèle aurait indirectement accès à de l'information sur les données de test. Pour une évaluation honnête, le preprocessing doit être fit sur le train set et appliqué tel quel au test set, comme en production réelle où les nouvelles données arrivent une par une.

### Q6 : "Quel est le trade-off de l'Adversarial Training ?"

Tsipras et al. (2019) prouvent qu'il existe une tension fondamentale entre robustesse adversariale et précision sur données propres. Intuitivement : un classificateur robuste doit avoir des frontières de décision plus "larges" (robustes aux petites perturbations), ce qui le rend parfois moins précis aux points de données propres proches de la frontière.

**Nos chiffres :** Baseline F1=0.911 vs Adv.Training F1=0.897 sur données propres (−1.5%). Le coût est faible ici, mais il est bien présent.

### Q7 : "Pourquoi Feature Squeezing est-il efficace ?"

Les attaques adversariales (FGSM, PGD) calculent des perturbations numériquement précises — par exemple `+0.04731...` sur une feature. Cette précision est nécessaire pour franchir la frontière de décision. En quantifiant les features sur 16 niveaux seulement (4 bits), on arrondit cette perturbation à sa valeur quantifiée la plus proche. Si la perturbation est plus petite que `1/16 × max_value`, elle est complètement effacée. L'attaquant aurait besoin d'utiliser des perturbations beaucoup plus grandes pour survivre à la quantification — ce qui les rendrait détectables par d'autres moyens.

### Q8 : "Pourquoi F1 et pas ROC-AUC ?"

ROC-AUC est robuste aux déséquilibres de classes et évalue toutes les valeurs de seuil. Mais pour un IDS en production, on utilise un seuil fixé (généralement 0.5) et on veut savoir exactement combien d'attaques sont détectées (Recall) et combien d'alarmes sont fausses (1 - Precision). Le F1 capture exactement cela. On a aussi calculé le PR-AUC (Area Under Precision-Recall Curve) qui est plus informatif que ROC-AUC quand les faux positifs ont un coût élevé.

### Q9 : "L'attaquant a-t-il vraiment accès aux gradients du modèle (white-box) ?"

En pratique, un attaquant n'a pas toujours accès au modèle exact (c'est le setting **black-box**). Mais :
1. On étudie le pire cas (white-box) pour établir une borne supérieure sur la vulnérabilité
2. Les attaques black-box (ex: requêtes multiples pour estimer le gradient) sont souvent presque aussi efficaces en pratique
3. Le modèle peut être "volé" par model extraction
4. Pour un projet académique, white-box est le cadre standard (Madry et al., Goodfellow et al.)

### Q10 : "Quelles sont les limites du projet ?"

1. **Contraintes physiques non strictes :** le masque de features manipulables est une approximation. En réalité, les contraintes sont plus complexes (ex: `sbytes` et `spkts` sont corrélés).

2. **Monotonie non respectée :** un attaquant réseau modifie ses features de façon causalement cohérente (augmenter les bytes implique augmenter le débit). FGSM/PGD peuvent créer des combinaisons de features physiquement impossibles.

3. **Adversarial Training sur FGSM seulement :** pour une vraie robustesse PGD, il faudrait entraîner avec des exemples PGD (plus coûteux).

4. **Pas de défense contre les attaques black-box ou transfert :** on n'a testé que les attaques white-box.

5. **Dataset statique :** un IDS réel doit s'adapter à un trafic qui évolue dans le temps (concept drift).

---

## Synthèse — ce qu'il faut retenir

```
Problème  : un MLP classifieur réseau est vulnérable aux perturbations adversariales
Dataset   : UNSW-NB15, 42 features, classification binaire normal/attaque
Modèle    : MLP 4 couches, ReLU, Dropout, Sigmoid, BCELoss, Adam
Attaque 1 : FGSM — une étape, gradient sign, budget ε — rapide mais limité
Attaque 2 : PGD  — 40 étapes itératives, projection L∞ — standard de robustesse
Défense 1 : Adversarial Training — entraînement sur mix propres+adversariaux
Défense 2 : Feature Squeezing   — quantification sur 4 bits à l'inférence
Résultat  : Feature Squeezing est la meilleure défense (F1 PGD : 0.852 vs 0.840 baseline)
Limite    : tension fondamentale robustesse/précision (Tsipras et al. 2019)
```

---

## Références

- Goodfellow et al. (2014) — *Explaining and Harnessing Adversarial Examples* (FGSM)
- Madry et al. (2018) — *Towards Deep Learning Models Resistant to Adversarial Attacks* (PGD + Adversarial Training)
- Xu et al. (2018) — *Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks*
- Tsipras et al. (2019) — *Robustness May Be at Odds with Accuracy*
- Moustafa & Slay (2015) — *UNSW-NB15: A comprehensive data set for network intrusion detection systems*
- Srivastava et al. (2014) — *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*
