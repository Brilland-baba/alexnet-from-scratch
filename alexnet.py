import math
import torch
import torch.nn as nn
import torch.nn.functional as F


#  Un fonction pour l'initialisation He : std = sqrt(2/fan_in), il va aider à compenser les 50% de neurones
#    annulés par ReLU pour maintenir la variance du signal couche après couche
def he_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        fan_in = m.weight.data[0].numel() if isinstance(m, nn.Conv2d) else m.in_features
        nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_in))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


#  Local Response Normalisation
#    Normalise chaque activation par ses n voisins sur l'axe canal :
#    b = a / (k + alpha * sum(a_j^2))^beta

class LRN(nn.Module):
    def __init__(self, n=5, k=2.0, alpha=1e-4, beta=0.75):
        super().__init__()
        self.n, self.k, self.alpha, self.beta = n, k, alpha, beta

    def forward(self, x):
        B, C, H, W = x.shape
        sq = (x ** 2).permute(0, 2, 3, 1).reshape(-1, 1, C)
        local = F.avg_pool1d(sq, self.n, stride=1, padding=self.n // 2) * self.n
        denom = (self.k + self.alpha * local.reshape(B, H, W, C).permute(0, 3, 1, 2)) ** self.beta
        return x / denom


# Notre Bloc convolutif modulaire : Conv → ReLU → Norm
#    norm="bn" | "lrn" | "none"  , le  seul paramètre qui varie dans l'ablation study

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s, p, norm="bn", pool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=(norm == "none"))
        self.act  = nn.ReLU(inplace=True)
        self.norm = {"lrn": LRN(), "bn": nn.BatchNorm2d(out_c)}.get(norm)
        self.pool = nn.MaxPool2d(2, 2) if pool else None

    def forward(self, x):
        x = self.act(self.conv(x))
        if self.norm: x = self.norm(x)
        if self.pool: x = self.pool(x)
        return x


# Le AlexNet complet : 5 ConvBlocks + classificateur FC3
#    Adapté sur CIFAR-10 64x64 : MaxPool 2x2, FC réduit (2048→1024), AdaptivePool(4,4)


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, norm="bn", p=0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   96,  11, 4, 2, norm, pool=True),
            ConvBlock(96,  256,  5, 1, 2, norm, pool=True),
            ConvBlock(256, 384,  3, 1, 1, "none"),
            ConvBlock(384, 384,  3, 1, 1, "none"),
            ConvBlock(384, 256,  3, 1, 1, "none", pool=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Dropout(p), nn.Linear(256*4*4, 2048), nn.ReLU(inplace=True),
            nn.Dropout(p), nn.Linear(2048, 1024),    nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
        self.apply(he_init)

    def forward(self, x):
        return self.head(torch.flatten(self.pool(self.features(x)), 1))
