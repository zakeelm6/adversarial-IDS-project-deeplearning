# Comprendre le Projet — Adversarially Robust IDS

> Ce document n'est pas un tutoriel de code.  
> C'est une explication conceptuelle et constructuelle de **pourquoi** on fait ce qu'on fait,  
> **comment** ça fonctionne en profondeur, et **ce que ça signifie** en termes de sécurité.

---

## 1. Le problème fondamental : pourquoi les IDS échouent

### 1.1 Ce qu'un IDS apprend vraiment

Un IDS basé sur du machine learning apprend une **frontière de décision** dans un espace de features.

Imagine un espace à 2 dimensions (simplifié) :

```
  durée
    ↑
    |  × × ×  ← attaques (courtes, beaucoup de paquets)
    |  × × ×
    |---------------- frontière apprise
    |  . . .  ← trafic normal (long, peu de paquets)
    |  . . .
    └──────────────→ nb_paquets
```

Le modèle apprend cette frontière sur des données d'entraînement.  
Le problème : **la frontière est apprise sur des attaques non-modifiées**.  
Elle ne correspond pas forcément à la vraie frontière entre "malveillant" et "bénin".

### 1.2 La faille conceptuelle

Le modèle apprend des **corrélations statistiques**, pas la **sémantique de l'attaque**.

Exemple concret avec un port scan :
- Ce qui définit un port scan : *tenter de se connecter à de nombreux ports*
- Ce que le modèle apprend réellement : *durée courte + beaucoup de paquets + ports séquentiels*

Ces deux choses ne sont pas équivalentes. Un attaquant peut maintenir la première (l'essence de l'attaque) tout en modifiant la seconde (les statistiques observées).

```
Essence de l'attaque       →  inchangée  →  l'attaque reste efficace
Statistiques observées     →  modifiées  →  le modèle ne la reconnaît plus
```

C'est la faille fondamentale. Le projet consiste à la comprendre et à la corriger.

---

## 2. Le Threat Model — raisonner comme un attaquant

### 2.1 Pourquoi définir un threat model avant de coder

Le threat model n'est pas une formalité administrative. Il répond à une question cruciale :

> **Contre quoi exactement cherche-t-on à être robuste ?**

Sans threat model, on ne sait pas quelles attaques générer, quelles features perturber, ni si les défenses qu'on implémente ont du sens. On risque de défendre contre des attaques irréalistes.

### 2.2 Les 4 dimensions du threat model

#### Dimension 1 — Connaissance de l'attaquant

| Scénario | Ce que l'attaquant sait | Réalisme |
|---|---|---|
| **White-box** | Architecture, poids, features | Cas académique / attaquant interne |
| **Gray-box** | Type de modèle, features utilisées | Cas probable (reverse engineering) |
| **Black-box** | Seulement les prédictions (oui/non) | Cas le plus réaliste |

On implémente d'abord **white-box** (FGSM/PGD) car c'est le cas le plus fort — si la défense résiste au pire cas, elle résiste aux autres. Mais on doit mentionner que le cas réel est souvent gray/black-box.

#### Dimension 2 — Capacité de manipulation

L'attaquant ne peut pas modifier n'importe quelle feature. Il y a des **contraintes physiques** :

```
Features réseau divisées en 3 catégories :

CONTRÔLABLES par l'attaquant :
  - Timing des paquets (inter_arrival_time)
  - Taille des paquets (sbytes, dbytes) ← padding possible
  - Nombre de paquets envoyés (spkts)
  - TTL (configurable dans la pile réseau)
  - Durée de la connexion (dur) ← contrôlé par l'attaquant

NON-CONTRÔLABLES sans casser l'attaque :
  - IP destination ← si on la change, l'attaque rate sa cible
  - Port destination ← la faille exploitée est sur un port précis
  - Protocole ← changer TCP en UDP change la nature de l'attaque

PARTIELLEMENT CONTRÔLABLES :
  - Flags TCP ← certains sont imposés par le protocole
  - Ports source ← libre, mais certains IDS surveillent les plages
```

#### Dimension 3 — Objectif de l'attaquant

Dans ce projet, l'objectif est **l'évasion** : faire classer une attaque comme trafic normal.  
Ce n'est pas la seule possibilité :

```
Évasion    → attaque classifiée comme normale         ← notre focus
Empoisonnement → corrompre l'entraînement du modèle   ← Projet 6 du cours
Inversion  → reconstruire les données d'entraînement  ← autre domaine
```

#### Dimension 4 — Contrainte de budget

L'attaquant ne peut pas perturber infiniment. Si la modification est trop grande :
- L'attaque devient détectable par d'autres moyens (logs, signatures)
- L'attaque cesse de fonctionner (trop lent, trop fragmenté)

On formalise ça par **epsilon (ε)** : la magnitude maximale de perturbation autorisée.

```
ε = 0.05  →  perturbation très faible, subtile
ε = 0.1   →  perturbation modérée (notre cas principal)
ε = 0.3   →  perturbation forte, potentiellement détectable
```

---

## 3. Les attaques adversariales — la mécanique

### 3.1 Ce que le gradient représente

Pour comprendre FGSM, il faut comprendre ce qu'est le gradient dans ce contexte.

Le modèle produit un score de confiance entre 0 et 1 :
```
score = 0.97  →  "je suis sûr à 97% que c'est une attaque"
score = 0.03  →  "je suis sûr à 97% que c'est du trafic normal"
```

La loss mesure à quel point le modèle se trompe.  
Le gradient de la loss par rapport aux features dit :

> *"Si j'augmente cette feature de 0.001, de combien la loss augmente-t-elle ?"*

Autrement dit, le gradient pointe dans la direction qui **maximise l'erreur du modèle**.

### 3.2 FGSM — Fast Gradient Sign Method

**Idée** : perturber les features dans la direction du gradient pour maximiser la loss.

```
Formule :
  X_adv = X + ε × sign(∇_X L(f(X), y))

Où :
  X          = features originales de l'attaque
  ε          = budget de perturbation
  ∇_X L      = gradient de la loss par rapport aux features
  sign(...)  = garde seulement le signe (+1 ou -1)
  X_adv      = exemple adversarial
```

En pratique, pour chaque feature manipulable :
```
si le gradient est positif  →  on ajoute ε   (augmente la feature)
si le gradient est négatif  →  on soustrait ε (diminue la feature)
```

**Limite de FGSM** : c'est une attaque en un seul pas. Elle n'est pas optimale.

### 3.3 PGD — Projected Gradient Descent

**Idée** : répéter FGSM en petits pas, en restant dans la boule de rayon ε.

```
Algorithme :
  X_0 = X (départ : exemple original)
  
  Pour t = 1 à T :
    X_t = X_{t-1} + α × sign(∇_X L(f(X_{t-1}), y))
    X_t = Projection(X_t, boule(X, ε))   ← on reste dans le budget

  X_adv = X_T
```

La **projection** garantit qu'on ne dépasse jamais le budget ε.  
Avec assez d'itérations, PGD trouve la perturbation **optimale** dans la boule ε.

```
Comparaison visuelle :

FGSM :  X ──────────────────────→ X_adv   (un seul grand pas)

PGD  :  X → → → → → → → → → → → X_adv   (beaucoup de petits pas)
             (chaque pas suit le gradient local)
```

**Pourquoi PGD est plus fort** : le gradient local change à chaque pas. PGD s'adapte, FGSM non.

### 3.4 La contrainte de réalisme — ce qui différencie ce projet

Dans la littérature classique (images), les perturbations sont appliquées à tous les pixels sans restriction. Ici, on ajoute une **contrainte de domaine** : ne perturber que les features manipulables.

```python
# Sans contrainte (irréaliste pour le réseau) :
X_adv = X + ε × sign(gradient)   # toutes les features perturbées

# Avec contrainte (notre approche) :
mask = [1, 0, 1, 1, 0, 0, 1, ...]   # 1 = manipulable, 0 = non-manipulable
X_adv = X + ε × sign(gradient) × mask
```

C'est ce qui rend le projet scientifiquement solide : les attaques générées correspondent à ce qu'un vrai attaquant pourrait faire.

---

## 4. Les défenses — la mécanique

### 4.1 Pourquoi le baseline échoue sous attaque

Le modèle baseline a appris une frontière sur des données propres.  
Sous attaque FGSM, les exemples adversariaux se retrouvent **de l'autre côté de la frontière** :

```
  durée
    ↑
    |  × × ×         ← attaques originales (côté attaque)
    |  · · ·         ← exemples adversariaux (passent côté normal !)
    |---------------- frontière apprise
    |  . . .
    └──────────────→ nb_paquets
```

Le problème : la frontière a des **zones de faible densité** où une petite perturbation suffit à changer la prédiction.

### 4.2 Adversarial Training — reconstruire la frontière

**Idée** : montrer au modèle pendant l'entraînement que les attaques ET leurs versions adversariales sont toutes des attaques.

```
Entraînement normal :
  modèle voit  [attaque originale]  →  apprend frontière étroite

Adversarial training :
  modèle voit  [attaque originale]
              +[version FGSM ε=0.05]
              +[version FGSM ε=0.1 ]   →  apprend frontière élargie
              +[version PGD        ]
```

Visuellement, la frontière s'élargit autour des attaques :

```
  durée
    ↑
    |  ×[   ×  ]×       ← [zone robuste] autour des attaques
    |  ×[   ×  ]×
    |---------------- nouvelle frontière (plus large)
    |  . . .
    └──────────────→ nb_paquets
```

**Le trade-off inévitable** :  
En élargissant la zone "attaque", on risque de classer certains trafics normaux comme attaques.  
C'est pourquoi les performances sur données propres baissent légèrement après adversarial training.  
Ce trade-off doit être mesuré et discuté dans le rapport.

### 4.3 Feature Squeezing — effacer les perturbations

**Idée** : réduire la précision des features pour que les petites perturbations disparaissent.

```
Attaquant calcule la perturbation optimale :
  durée_original  = 0.00100000
  perturbation    = +0.00183746   ← calculée précisément par FGSM
  durée_adversarial = 0.00283746

Après feature squeezing (arrondi à 2 décimales) :
  durée_squeezed = 0.00           ← la perturbation est effacée

Le modèle voit maintenant 0.00 au lieu de 0.00283746
→ La perturbation n'a plus d'effet
```

**Limite** : si la perturbation est grande (ε élevé), l'arrondi ne suffit plus.  
Feature squeezing est efficace contre les petites perturbations, moins contre PGD fort.

### 4.4 Pourquoi comparer les deux défenses

Les deux défenses ont des profils différents :

```
                      Coût computationnel   Robustesse FGSM   Robustesse PGD
Adversarial Training  Élevé (2x entraîne.)  Haute             Haute
Feature Squeezing     Faible (post-proc.)   Moyenne           Faible
```

Aucune n'est universellement meilleure. C'est pour ça qu'on les compare : pour montrer dans quels cas chacune est préférable, et justifier le choix selon les contraintes (temps, ressources, type d'attaque attendu).

---

## 5. Les métriques — pourquoi pas l'accuracy

### 5.1 Le problème du déséquilibre

Dans UNSW-NB15 :
```
Trafic normal : ~56 000 exemples  (32%)
Attaques      : ~119 000 exemples (68%)
```

Un modèle qui dit **toujours "attaque"** aurait une accuracy de 68%.  
Ce modèle est inutile. L'accuracy ne mesure pas ce qui compte.

### 5.2 Ce que chaque métrique mesure

```
Precision  =  Parmi les alarmes déclenchées, combien sont vraies ?
              → mesure le taux de faux positifs (fausses alertes)

Recall     =  Parmi toutes les vraies attaques, combien a-t-on détectées ?
              → mesure le taux de faux négatifs (attaques manquées)

F1-score   =  Moyenne harmonique de Precision et Recall
              → équilibre les deux

PR-AUC     =  Aire sous la courbe Precision-Recall
              → mesure la robustesse sur tous les seuils de décision
```

### 5.3 L'Evasion Rate — métrique spécifique à ce projet

```
Evasion Rate = nb d'attaques classifiées comme normales sous attaque adversariale
               ─────────────────────────────────────────────────────────────────
               nb total d'attaques

Interprétation :
  Evasion Rate = 60%  →  l'attaquant réussit à se cacher 60% du temps
  Evasion Rate = 10%  →  le modèle reste robuste, détecte 90% même sous attaque
```

L'objectif du projet est de **réduire l'evasion rate** avec les défenses.

---

## 6. Ce qu'on construit — vue d'ensemble

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE COMPLET                         │
│                                                             │
│  [UNSW-NB15 CSV]                                            │
│       ↓                                                     │
│  [Preprocessing]  ← nettoyage, normalisation, split         │
│       ↓                                                     │
│  [Baseline MLP]   ← 3-4 couches, ReLU, Dropout              │
│       ↓                                                     │
│  [Évaluation 1]   ← F1, PR-AUC sur données propres          │
│       ↓                                                     │
│  [FGSM Attack]  [PGD Attack]   ← perturbations contraintes  │
│       ↓               ↓                                     │
│  [Évaluation 2]   ← F1, Evasion Rate sous attaque           │
│       ↓                                                     │
│  [Adv. Training] [Feature Squeezing]   ← défenses           │
│       ↓                 ↓                                   │
│  [Évaluation 3]   ← comparaison robustesse finale           │
│       ↓                                                     │
│  [Rapport]  ← analyse, conclusions, trade-offs              │
└─────────────────────────────────────────────────────────────┘
```

### Les 3 variations expérimentales requises (et leur sens)

**Variation 1 — Budget d'attaque (ε = 0.05, 0.1, 0.3)**  
Question sous-jacente : *à partir de quel budget la défense commence-t-elle à échouer ?*  
On trace une courbe Robustesse = f(ε) pour chaque modèle.

**Variation 2 — Architecture (MLP 2 couches vs 4 couches)**  
Question sous-jacente : *est-ce qu'un modèle plus complexe est naturellement plus robuste ?*  
Réponse attendue : non, la complexité n'implique pas la robustesse. Ça doit être appris.

**Variation 3 — Défense (Adversarial Training vs Feature Squeezing)**  
Question sous-jacente : *quel mécanisme de défense est le plus efficace et dans quel contexte ?*  
On compare sur FGSM faible, FGSM fort, PGD.

---

## 7. Ce que le rapport doit démontrer

Le rapport n'est pas un compte-rendu de ce qu'on a fait.  
C'est une **argumentation scientifique** qui répond à :

1. **Le problème est-il réel ?**  
   → Montrer que le baseline se fait tromper (evasion rate élevé sous attaque)

2. **Les attaques sont-elles réalistes ?**  
   → Justifier le threat model et les features manipulables

3. **Les défenses fonctionnent-elles ?**  
   → Montrer la réduction de l'evasion rate avec les défenses

4. **À quel prix ?**  
   → Montrer et discuter le trade-off robustesse / performance normale

5. **Quelles sont les limites ?**  
   → Discuter ce que les défenses ne couvrent pas (ex : black-box, ε très grand)

---

## 8. Concepts clés à maîtriser pour la soutenance

| Concept | Ce qu'il faut savoir expliquer |
|---|---|
| **Gradient** | C'est la direction dans laquelle la loss augmente le plus vite — FGSM l'utilise pour trouver la perturbation optimale |
| **Boule epsilon** | L'ensemble de toutes les perturbations de magnitude ≤ ε — PGD reste toujours dedans |
| **Trade-off robustesse/accuracy** | Un modèle robuste voit plus d'exemples difficiles → il apprend une frontière plus large → parfois trop large pour certains cas normaux |
| **Evasion vs empoisonnement** | Évasion = tromper le modèle au moment de la prédiction. Empoisonnement = corrompre les données d'entraînement |
| **Pourquoi l'accuracy est trompeuse** | À cause du déséquilibre de classes — un modèle qui dit toujours "attaque" a une bonne accuracy mais ne sert à rien |
| **White-box vs black-box** | White-box nécessite le gradient → accès au modèle. Black-box utilise des requêtes pour estimer le gradient |
