import json, time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from alexnet import AlexNet



# Comme double  exécution (en local et avec Collab), si le  GPU disponible, on l'utilise. Sinon CPU.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# Les hyperparamètres.
CFG = dict(bs=256, epochs=50, lr=0.01, wd=1e-3, nc=10, val_ratio=0.1)
mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

# Augmentation des données (On vise un meilleur accuracy): RandomCrop + ColorJitter
T_train = transforms.Compose([
    transforms.Resize(72),
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(.3, .3, .2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
T_val = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

full   = datasets.CIFAR10("./data", train=True,  download=True, transform=T_train)
test_d = datasets.CIFAR10("./data", train=False, download=True, transform=T_val)
n_val  = int(len(full) * CFG["val_ratio"])
train_d, val_d = random_split(full, [len(full)-n_val, n_val],
                               generator=torch.Generator().manual_seed(42))

train_dl = DataLoader(train_d, CFG["bs"], shuffle=True,  num_workers=0, pin_memory=False)
val_dl   = DataLoader(val_d,   CFG["bs"], shuffle=False, num_workers=0, pin_memory=False)
test_dl  = DataLoader(test_d,  CFG["bs"], shuffle=False, num_workers=0, pin_memory=False)


def run_epoch(model, loader, crit, opt=None, sched=None):
    model.train() if opt else model.eval()
    loss_sum = correct = n = 0
    ctx = torch.enable_grad() if opt else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X); loss = crit(out, y)
            if opt:
                opt.zero_grad(); loss.backward(); opt.step()
                if sched: sched.step()
            loss_sum += loss.item() * len(X)
            correct  += (out.argmax(1) == y).sum().item()
            n        += len(X)
    return loss_sum / n, correct / n



# Entraînement d'une variante puis sauvegarde du meilleur checkpoint

def run(norm_tag):
    model = AlexNet(CFG["nc"], norm=norm_tag).to(DEVICE)
    crit  = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt   = optim.SGD(model.parameters(), CFG["lr"], momentum=.9,
                      weight_decay=CFG["wd"], nesterov=True)
    # OneCycleLR : on parcours le lr rapidement puis on  descend . À chaque epoch, on va entraîner puis on valide
    sched = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=CFG["lr"],
        epochs=CFG["epochs"], steps_per_epoch=len(train_dl),
        pct_start=0.3, div_factor=10, final_div_factor=100,
    )
    best = 0.0

    print(f"\n  norm={norm_tag}  |  device={DEVICE}")
    for ep in range(1, CFG["epochs"] + 1):
        t = time.time()
        tl, ta = run_epoch(model, train_dl, crit, opt, sched)
        vl, va = run_epoch(model, val_dl, crit)
        if va > best: best = va; torch.save(model.state_dict(), f"ckpt_{norm_tag}.pt")
        print(f"  ep{ep:02d}  tr={ta:.3f}  val={va:.3f}  [{time.time()-t:.1f}s]")

    model.load_state_dict(torch.load(f"ckpt_{norm_tag}.pt", map_location=DEVICE))
    _, test_acc = run_epoch(model, test_dl, crit)
    print(f"  → test_acc={test_acc*100:.2f}%")
    return best, test_acc


# On entraîne les 3 variantes dans l'ordre. À la fin, on sait quelle variante performe mieux et on affiche un résumé des résultats.


if __name__ == "__main__":
    results = {}
    for tag in ["bn", "lrn", "none"]:
        bv, ta = run(tag)
        results[tag] = dict(best_val=round(bv, 4), test_acc=round(ta, 4))

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n Ablation métriques  Summary")
    for t, r in results.items():
        print(f"  {t:4s}  val={r['best_val']*100:.1f}%  test={r['test_acc']*100:.1f}%")
