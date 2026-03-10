# AlexNet — From Scratch (PyTorch)

**TP n°2 Advanced Deep Learning · ENEAM/ISE**  
Étudiant : BABA C.F. Brilland | Enseignant : R.O.Y. TOHA

---

## Résultats — CIFAR-10 (50 epochs, CPU)

| Variante | Val Acc | Test Acc |
|----------|---------|----------|
| **BatchNorm** | **80.5%** | **81.3%** |
| None (baseline) | 80.0% | 80.7% |
| LRN (original) | 79.5% | 80.2% |

> **Résultat clé :** None > LRN — LRN conçue pour ImageNet 224×224 pénalise les feature maps réduites (64×64).

## Architecture

5 blocs convolutifs modulaires (`ConvBlock`) + classificateur FC3.  
`Conv2d → ReLU → Norm → MaxPool 2×2` (optionnel).  
LRN implémentée manuellement (formule papier). Init He sur toutes les couches.

| Bloc | Canaux | Kernel | Norm |
|------|--------|--------|------|
| C1 | 3→96 | 11×11/4 | ablation |
| C2 | 96→256 | 5×5/1 | ablation |
| C3–C5 | 256→384→256 | 3×3/1 | — |
| FC | 4096→2048→1024→10 | — | Dropout 0.5 |

## Setup & Run

```bash
pip install torch torchvision matplotlib
python train.py       # entraîne 3 variantes → results.json + ckpt_*.pt
python visualize.py   # génère ablation.png
```

**Test rapide (reproductibilité) :**
```bash
python -c "from alexnet import AlexNet; import torch; m=AlexNet(10,'bn'); print(m(torch.randn(2,3,64,64)).shape)"
# → torch.Size([2, 10])
```

## Structure

```
├── alexnet.py       # LRN, ConvBlock, AlexNet, init He
├── train.py         # entraînement + ablation study
├── visualize.py     # courbes + bar chart
├── results.json     # métriques finales
└── requirements.txt
```
