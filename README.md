# AlexNet — From Scratch (PyTorch)

**TP n°2 · Advanced Deep Learning · UAC/ENEAM/ISE**  
Étudiant : BABA C.F. Brilland | Enseignant :Rodéo Oswald Y. TOHA (Engineer in CV & GenAI)

---

## Contexte & Intuition

L'objectif de ce projet est de **reconstruire AlexNet intégralement from scratch** — sans modèle pré-entraîné, sans boîte noire — en partant du papier original de Krizhevsky, Sutskever & Hinton (NeurIPS 2012).

AlexNet est le premier CNN profond à avoir dominé ImageNet (top-5 error 15.3%), marquant le début de l'ère deep learning en vision. Le reconstruire from scratch permet de comprendre chaque choix architectural : pourquoi ReLU au lieu de sigmoïde, pourquoi la LRN, pourquoi ces tailles de filtres. Ce n'est pas juste du code — c'est lire un papier et le traduire en mathématiques exécutables.

L'adaptation à **CIFAR-10 64×64** (au lieu d'ImageNet 224×224) impose des choix techniques justifiés : réduction des MaxPool, adaptation du classifieur, ce qui constitue en soi une contribution architecturale.

---

## Ce que j'ai implémenté & comment

### 1. Local Response Normalisation (LRN) — from scratch

La LRN du papier (eq. 3) n'utilise pas `nn.LocalResponseNorm`. Elle est réimplémentée manuellement :

```
b_x^i = a_x^i / ( k + alpha * sum_{j=max(0,i-n/2)}^{min(N-1,i+n/2)} (a_x^j)^2 )^beta
```

Implémentation via `F.avg_pool1d` sur l'axe canal après reshape — la seule façon de faire une fenêtre glissante sur les canaux en PyTorch sans boucle.

### 2. ConvBlock modulaire

Chaque bloc encapsule `Conv2d → ReLU → Norm → MaxPool` avec un seul paramètre `norm="bn"|"lrn"|"none"`. C'est ce paramètre unique qui permet l'ablation study : on change une seule chose, tout le reste est identique.

### 3. Initialisation He (Kaiming 2015)

ReLU annule ~50% des activations. La variance de sortie d'une couche est :

```
Var[z_l] = fan_in · Var[W] · E[ReLU(x)²] = (fan_in · Var[W]) / 2
```

Pour maintenir `Var[z_l] = Var[z_{l-1}]` : **std = sqrt(2 / fan_in)**.  
Xavier (`sqrt(2/(fan_in+fan_out))`) ignore ce facteur ½ → variance s'effondre en profondeur.  
Vérifié expérimentalement : ratio `actual_std / expected_std = 1.00` sur les 8 couches.

### 4. Pipeline d'entraînement

- **SGD Nesterov** : anticipe la direction du gradient → convergence plus propre
- **OneCycleLR** : monte lr 0.001→0.01 sur 30% des epochs puis descend → escape local minima
- **Label smoothing 0.1** : évite la sur-confiance du modèle sur les classes
- **RandomCrop(64, padding=4)** : augmentation spatiale forte, +4-6% accuracy sur CIFAR

---

## Résultats — CIFAR-10 (50 epochs, CPU)

| Variante | Val Acc | Test Acc |
|----------|---------|----------|
| **BatchNorm** | **80.5%** | **81.3%** |
| None (baseline) | 80.0% | 80.7% |
| LRN (original) | 79.5% | 80.2% |

### Résultat clé de l'ablation study

**None (80.7%) > LRN (80.2%)** — résultat contre-intuitif et scientifiquement valide.

LRN est conçue pour des feature maps larges (ImageNet 224×224) où la compétition inter-canaux a du sens. À 64×64, après C1 (stride=4) et deux MaxPool(2,2), les feature maps font ~4×4 en C3 : la fenêtre de n=5 canaux compétiteurs inhibe des activations utiles sans gain de spécialisation. BN normalise sur l'axe mini-batch indépendamment de la résolution — toujours efficace.

---

## Architecture

5 blocs convolutifs modulaires (`ConvBlock`) + classificateur FC3.  
Pipeline : `Conv2d → ReLU → Norm → MaxPool 2×2` (optionnel par bloc).

| Bloc | Canaux | Kernel/Stride | Norm | Pool |
|------|--------|--------------|------|------|
| C1 | 3 → 96 | 11×11 / 4 | ablation | 2×2/2 |
| C2 | 96 → 256 | 5×5 / 1 | ablation | 2×2/2 |
| C3 | 256 → 384 | 3×3 / 1 | — | — |
| C4 | 384 → 384 | 3×3 / 1 | — | — |
| C5 | 384 → 256 | 3×3 / 1 | — | 2×2/2 |
| FC6 | 9 216 → 2 048 | — | Dropout 0.5 | — |
| FC7 | 2 048 → 1 024 | — | Dropout 0.5 | — |
| FC8 | 1 024 → 10 | — | — | — |

**14 246 634 paramètres** (vs 60M original — FC réduit pour éviter le surapprentissage sur CIFAR-10).

---

## Setup & Run

```bash
pip install torch torchvision matplotlib

# Entraîne les 3 variantes → results.json + ckpt_*.pt
python train.py

# Génère ablation.png (courbes + bar chart)
python Plots.py
```

**Test rapide reproductibilité :**
```bash
python -c "from alexnet import AlexNet; import torch; m=AlexNet(10,'bn'); print(m(torch.randn(2,3,64,64)).shape)"
# → torch.Size([2, 10])
```

**Vérification init He :**
```bash
python -c "
import torch, math
from alexnet import AlexNet
m = AlexNet(10, 'bn')
for name, p in m.named_parameters():
    if 'weight' in name and p.dim() > 1:
        fan_in = p.data[0].numel()
        ratio = p.data.std().item() / math.sqrt(2.0 / fan_in)
        print(f'{name:40s}  ratio={ratio:.2f}')
"
# → ratio ≈ 1.00 sur  les couches
```

---

## Structure du projet

```
├── alexnet.py       # LRN manuelle, ConvBlock, AlexNet, init He
├── train.py         # entraînement + ablation study (3 variantes)
├── Plots.py     # courbes val/train + bar chart ablation
├── results.json     # métriques finales (val/test accuracy par variante)
└── requirements.txt
```

---

## Références consultées 

- Krizhevsky, A., Sutskever, I., & Hinton, G.E. (2012). *ImageNet Classification with Deep Convolutional Neural Networks*. NeurIPS.
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Delving Deep into Rectifiers*. ICCV.
