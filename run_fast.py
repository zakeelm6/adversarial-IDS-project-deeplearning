import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              average_precision_score, confusion_matrix, ConfusionMatrixDisplay)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Optimisation CPU
torch.set_num_threads(2)

print("=== Chargement donnees ===")

def preprocess(tr, te):
    for df in [tr, te]:
        df.drop(columns=[c for c in ['id','attack_cat'] if c in df.columns], inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    med = tr.median(numeric_only=True)
    tr.fillna(med, inplace=True); te.fillna(med, inplace=True)
    cat = [c for c in tr.select_dtypes(include=['object','str']).columns if c != 'label']
    le = LabelEncoder()
    for col in cat:
        tr[col] = le.fit_transform(tr[col].astype(str))
        kn = set(le.classes_)
        te[col] = le.transform(te[col].astype(str).apply(lambda x: x if x in kn else le.classes_[0]))
    sc = StandardScaler()
    Xtr = sc.fit_transform(tr.drop('label', axis=1).values.astype(np.float32))
    ytr = tr['label'].values.astype(np.float32)
    Xte = sc.transform(te.drop('label', axis=1).values.astype(np.float32))
    yte = te['label'].values.astype(np.float32)
    return Xtr, ytr, Xte, yte, sc

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

tr = pd.read_csv('data/UNSW_NB15_training-set.csv')
te = pd.read_csv('data/UNSW_NB15_testing-set.csv')
Xtr, ytr, Xte, yte, sc = preprocess(tr.copy(), te.copy())
print(f"Train: {Xtr.shape} | Test: {Xte.shape}")

mb = MLP(Xtr.shape[1])
mb.load_state_dict(torch.load('results/baseline_model.pth', weights_only=True))
mb.eval()

cols = pd.read_csv('data/UNSW_NB15_training-set.csv').drop(columns=['id','attack_cat','label']).columns.tolist()
MANIP = ['dur','spkts','dpkts','sbytes','dbytes','sttl','sload','dload','sinpkt','dinpkt','sjit','djit','smean','dmean']
midx = [cols.index(f) for f in MANIP if f in cols]
mask = torch.zeros(len(cols)); mask[midx] = 1.0

# Sous-ensemble pour les attaques (plus rapide)
aidx = np.where(yte == 1)[0]
Xa = Xte[aidx]; ya = yte[aidx]
EPS = 0.1

def fgsm(m, X, y, eps):
    Xt = torch.FloatTensor(X).requires_grad_(True)
    nn.BCELoss()(m(Xt).squeeze(), torch.FloatTensor(y)).backward()
    return (Xt + eps * Xt.grad.sign() * mask).detach().numpy()

def pgd(m, X, y, eps, steps=10):  # steps reduit : 40 -> 10
    Xo = torch.FloatTensor(X); Xd = Xo.clone(); yt = torch.FloatTensor(y)
    alpha = eps / steps
    for _ in range(steps):
        Xd = Xd.detach().requires_grad_(True)
        nn.BCELoss()(m(Xd).squeeze(), yt).backward()
        Xd = Xo + torch.clamp(Xd + alpha * Xd.grad.sign() * mask - Xo, -eps, eps)
    return Xd.detach().numpy()

def fsq(X, b=4):
    mv = np.max(np.abs(X), axis=0, keepdims=True) + 1e-8
    return np.round(X / mv * (2**b)) / (2**b) * mv

def evl(m, X, y):
    m.eval()
    with torch.no_grad():
        prob = m(torch.FloatTensor(X)).squeeze().numpy()
        pred = (prob >= 0.5).astype(int)
    return {
        'F1': f1_score(y, pred), 'Precision': precision_score(y, pred),
        'Recall': recall_score(y, pred), 'PR-AUC': average_precision_score(y, prob),
        'Evasion%': (pred[y == 1] == 0).mean() * 100, 'pred': pred
    }

# Adversarial Training rapide : 10 epochs, batch 1024
print("\n=== Adversarial Training (rapide) ===")
ma = MLP(Xtr.shape[1])
opt = torch.optim.Adam(ma.parameters(), lr=0.001)
crit = nn.BCELoss()
EPOCHS, BATCH = 10, 1024  # reduit : 30->10 epochs, 512->1024 batch

for ep in range(EPOCHS):
    ma.train()
    Xadv = fgsm(ma, Xtr, ytr, EPS)
    Xm = np.concatenate([Xtr, Xadv]); ym = np.concatenate([ytr, ytr])
    idx = np.random.permutation(len(Xm))
    Xt2 = torch.FloatTensor(Xm[idx]); yt2 = torch.FloatTensor(ym[idx])
    el = 0
    for s in range(0, len(Xt2), BATCH):
        xb = Xt2[s:s+BATCH]; yb = yt2[s:s+BATCH]
        opt.zero_grad()
        l = crit(ma(xb).squeeze(), yb); l.backward(); opt.step()
        el += l.item()
    print(f"  Epoch {ep+1}/{EPOCHS} | Loss: {el/(len(Xt2)//BATCH):.4f}")

torch.save(ma.state_dict(), 'results/adversarial_model.pth')
print("adversarial_model.pth OK")

# Generer les variantes adversariales
print("\n=== Generation des attaques ===")
Xfb = Xte.copy(); Xfb[aidx] = fgsm(mb, Xa, ya, EPS)
Xpb = Xte.copy(); Xpb[aidx] = pgd(mb, Xa, ya, EPS)
Xfa = Xte.copy(); Xfa[aidx] = fgsm(ma, Xa, ya, EPS)
Xpa = Xte.copy(); Xpa[aidx] = pgd(ma, Xa, ya, EPS)
Xsq = fsq(Xte); Xfsq = fsq(Xfb); Xpsq = fsq(Xpb)
print("Attaques generees OK")

# Resultats
rows = [
    ('Baseline',       'Propres', evl(mb, Xte,  yte)),
    ('Baseline',       'FGSM',    evl(mb, Xfb,  yte)),
    ('Baseline',       'PGD',     evl(mb, Xpb,  yte)),
    ('Adv.Training',   'Propres', evl(ma, Xte,  yte)),
    ('Adv.Training',   'FGSM',    evl(ma, Xfa,  yte)),
    ('Adv.Training',   'PGD',     evl(ma, Xpa,  yte)),
    ('Feat.Squeeze',   'Propres', evl(mb, Xsq,  yte)),
    ('Feat.Squeeze',   'FGSM',    evl(mb, Xfsq, yte)),
    ('Feat.Squeeze',   'PGD',     evl(mb, Xpsq, yte)),
]

print("\n=== RESULTATS FINAUX ===")
print(f"{'Modele':<16} {'Scenario':<10} {'F1':>7} {'PR-AUC':>8} {'Evasion%':>10}")
print("-" * 55)
for m, s, r in rows:
    print(f"{m:<16} {s:<10} {r['F1']:>7.4f} {r['PR-AUC']:>8.4f} {r['Evasion%']:>9.1f}%")

pd.DataFrame([{'Modele': m, 'Scenario': s, 'F1': r['F1'], 'Precision': r['Precision'],
               'Recall': r['Recall'], 'PR-AUC': r['PR-AUC'], 'Evasion%': r['Evasion%']}
              for m, s, r in rows]).to_csv('results/final_results.csv', index=False)

# Graphiques
print("\n=== Generation graphiques ===")
sc2 = ['Propres', 'FGSM', 'PGD']
mn  = ['Baseline', 'Adv.Training', 'Feat.Squeeze']
cl  = ['steelblue', 'tomato', 'seagreen']
x   = np.arange(3); w = 0.25

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
for i, (n, c) in enumerate(zip(mn, cl)):
    vf = [r['F1']       for m, s, r in rows if m == n]
    ve = [r['Evasion%'] for m, s, r in rows if m == n]
    b1 = a1.bar(x+(i-1)*w, vf, w, label=n, color=c)
    b2 = a2.bar(x+(i-1)*w, ve, w, label=n, color=c)
    a1.bar_label(b1, fmt='%.2f', padding=2, fontsize=8)
    a2.bar_label(b2, fmt='%.1f%%', padding=2, fontsize=8)
a1.set_title('F1-score'); a1.set_xticks(x); a1.set_xticklabels(sc2); a1.legend(); a1.set_ylim(0, 1.1)
a2.set_title('Evasion Rate (%)'); a2.set_xticks(x); a2.set_xticklabels(sc2); a2.legend()
plt.tight_layout(); plt.savefig('results/final_comparison.png', dpi=150); plt.close()
print("final_comparison.png OK")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
cfgs = [(mb, Xte, 'Baseline/Propres'), (mb, Xfb, 'Baseline/FGSM'), (mb, Xpb, 'Baseline/PGD'),
        (ma, Xte, 'Adv.Train/Propres'), (ma, Xfa, 'Adv.Train/FGSM'), (ma, Xpa, 'Adv.Train/PGD')]
for ax, (mdl, X, t) in zip(axes.flatten(), cfgs):
    with torch.no_grad():
        pred = (mdl(torch.FloatTensor(X)).squeeze().numpy() >= 0.5).astype(int)
    ConfusionMatrixDisplay(confusion_matrix(yte, pred), display_labels=['Normal', 'Attaque']).plot(
        ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(t, fontsize=9)
plt.tight_layout(); plt.savefig('results/confusion_matrices.png', dpi=150); plt.close()
print("confusion_matrices.png OK")

eps_list = [0.05, 0.1, 0.3]  # reduit : 5 -> 3 points
f1b, f1a, evb, eva = [], [], [], []
for eps in eps_list:
    Xp = Xte.copy(); Xp[aidx] = pgd(mb, Xa, ya, eps)
    r = evl(mb, Xp, yte); f1b.append(r['F1']); evb.append(r['Evasion%'])
    Xp = Xte.copy(); Xp[aidx] = pgd(ma, Xa, ya, eps)
    r = evl(ma, Xp, yte); f1a.append(r['F1']); eva.append(r['Evasion%'])
    print(f"  eps={eps} | Baseline F1={f1b[-1]:.3f} | Adv F1={f1a[-1]:.3f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
a1.plot(eps_list, f1b, 'o-', color='steelblue', label='Baseline')
a1.plot(eps_list, f1a, 's-', color='tomato', label='Adv.Training')
a1.set_title('F1 vs epsilon (PGD)'); a1.set_xlabel('Epsilon'); a1.legend()
a2.plot(eps_list, evb, 'o-', color='steelblue', label='Baseline')
a2.plot(eps_list, eva, 's-', color='tomato', label='Adv.Training')
a2.set_title('Evasion% vs epsilon (PGD)'); a2.set_xlabel('Epsilon'); a2.legend()
plt.tight_layout(); plt.savefig('results/robustness_vs_epsilon.png', dpi=150); plt.close()
print("robustness_vs_epsilon.png OK")

import os
print("\n=== Fichiers generes ===")
for f in sorted(os.listdir('results/')):
    print(f"  {f} ({os.path.getsize('results/'+f)//1024}KB)")
print("\nDONE")
