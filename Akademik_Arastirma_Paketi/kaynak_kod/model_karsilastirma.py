# -*- coding: utf-8 -*-
"""
model_karsilastirma.py — Mimari Ablasyon / Model Karşılaştırması
================================================================

Kayıtlı checkpoint'ler AYNI bağımsız test setinde karşılaştırılır. Her model
KENDİ mimarisi ve KENDİ eğitim ön işlemesiyle yüklenir — böylece karşılaştırma
adildir (elmayla elma). Bu, "hangi omurga (backbone) + eğitim stratejisi daha
iyi genelliyor?" sorusunu yanıtlayan bir mimari ablasyon çalışmasıdır.

Karşılaştırılan modeller (2.5D, orta 3 kesit, ikili sınıflandırma):
  - best_model_balanced_crop74      : ConvNeXt-Tiny + Focal Loss + sınıf dengesi
                                       + CenterCrop(280)  [ANA MODEL]
  - best_convnext_tiny_2.5d_focal    : ConvNeXt-Tiny + Focal Loss (önceki sürüm)
  - efficientnet_b0_best_model       : EfficientNet-B0 + CrossEntropy (Resize-224)
  - resnet50_2.5d_transfer_learning  : ResNet50 + CrossEntropy (Resize-224)

Not: MONAI DenseNet-3D (tek kanal, tam hacim) ve R(2+1)D (video) farklı GİRİŞ
biçimleri kullandığından bu birleşik 2.5D karşılaştırmasına dahil edilmemiştir;
dokümanda gelecek çalışma olarak not edilmiştir.

Çıktı:
  - sonuclar/model_karsilastirma.csv
  - sonuclar/model_karsilastirma.png

Kullanım:
    python model_karsilastirma.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
)

import ortak


# ============================================================
# Mimari kurucular (her checkpoint'in eğitim başıyla birebir)
# ============================================================
def build_convnext():
    m = models.convnext_tiny(weights=None)
    n = m.classifier[2].in_features
    m.classifier[2] = nn.Sequential(nn.Dropout(0.5), nn.Linear(n, 2))
    return m


def build_efficientnet():
    m = models.efficientnet_b0(weights=None)
    n = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(n, 256), nn.ReLU(),
        nn.Dropout(p=0.3), nn.Linear(256, 2),
    )
    return m


def build_resnet50():
    m = models.resnet50(weights=None)
    n = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Linear(n, 256), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(256, 2),
    )
    return m


# Ön işleme zincirleri
CONVNEXT_TF = ortak.build_transforms()   # CenterCrop(280)+Resize(224)+Normalize
PLAIN_TF = transforms.Compose([          # Resize(224)+Normalize (CenterCrop yok)
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=ortak.IMAGENET_MEAN, std=ortak.IMAGENET_STD),
])


# Model kayıt defteri
MODELLER = [
    {"ad": "ConvNeXt-Tiny (Focal+Dengeli+Crop) [ANA]",
     "dosya": "best_model_balanced_crop74.pth", "kur": build_convnext, "tf": CONVNEXT_TF},
    {"ad": "ConvNeXt-Tiny (Focal, önceki)",
     "dosya": "best_convnext_tiny_2.5d_focal.pth", "kur": build_convnext, "tf": CONVNEXT_TF},
    {"ad": "EfficientNet-B0 (CrossEntropy)",
     "dosya": "efficientnet_b0_best_model.pth", "kur": build_efficientnet, "tf": PLAIN_TF},
    {"ad": "ResNet50 (Transfer, CrossEntropy)",
     "dosya": "resnet50_2.5d_transfer_learning.pth", "kur": build_resnet50, "tf": PLAIN_TF},
]


@torch.no_grad()
def degerlendir(model, transform, split_dir, device, batch_size=16):
    ds = ortak.Medical25DDataset(root_dir=split_dir, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    yt, yp, ypr = [], [], []
    model.eval()
    for x, y in loader:
        x = x.to(device)
        probs = torch.softmax(model(x), dim=1)
        yt.extend(y.numpy().tolist())
        yp.extend(torch.argmax(probs, 1).cpu().numpy().tolist())
        ypr.extend(probs[:, 1].cpu().numpy().tolist())
    return np.array(yt), np.array(yp), np.array(ypr)


def metrikler(yt, yp, ypr):
    acc = accuracy_score(yt, yp)
    f1m = f1_score(yt, yp, average="macro")
    try:
        auc = roc_auc_score(yt, ypr)
    except ValueError:
        auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {"Doğruluk": acc, "Makro-F1": f1m, "ROC-AUC": auc,
            "Duyarlılık": sens, "Özgüllük": spec}


def main():
    ortak.SONUCLAR_DIR.mkdir(parents=True, exist_ok=True)
    device = ortak.get_device()
    print(f"[*] Cihaz: {device}")
    print(f"[*] Test seti: {ortak.TEST_DIR}\n")

    satirlar = []
    for cfg in MODELLER:
        yol = ortak.MODEL_DIR / cfg["dosya"]
        if not yol.exists():
            print(f"[-] Atlandı (bulunamadı): {cfg['dosya']}")
            continue
        try:
            model = cfg["kur"]().to(device)
            state = torch.load(str(yol), map_location=device, weights_only=True)
            model.load_state_dict(state)
        except Exception as e:
            print(f"[-] Yüklenemedi ({cfg['dosya']}): {e}")
            continue

        yt, yp, ypr = degerlendir(model, cfg["tf"], ortak.TEST_DIR, device)
        m = metrikler(yt, yp, ypr)
        m_row = {"Model": cfg["ad"], **m}
        satirlar.append(m_row)
        print(f"[+] {cfg['ad']:42s} | Acc={m['Doğruluk']*100:5.2f}%  "
              f"F1={m['Makro-F1']:.3f}  AUC={m['ROC-AUC']:.3f}  "
              f"Sens={m['Duyarlılık']:.3f}  Spec={m['Özgüllük']:.3f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not satirlar:
        print("[-] Hiçbir model değerlendirilemedi.")
        return

    df = pd.DataFrame(satirlar)
    csv_yol = ortak.SONUCLAR_DIR / "model_karsilastirma.csv"
    df.to_csv(csv_yol, index=False, encoding="utf-8-sig", float_format="%.4f")
    print(f"\n[+] Tablo kaydedildi: {csv_yol}")

    # --- Grup çubuk grafik (Doğruluk, Makro-F1, ROC-AUC) ---
    metrik_adlari = ["Doğruluk", "Makro-F1", "ROC-AUC"]
    x = np.arange(len(df))
    w = 0.25
    plt.figure(figsize=(12, 6), facecolor="white")
    renkler = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    for i, mad in enumerate(metrik_adlari):
        plt.bar(x + (i - 1) * w, df[mad].values, width=w, label=mad, color=renkler[i])
    plt.xticks(x, [a.replace(" ", "\n", 1) for a in df["Model"]], fontsize=8)
    plt.ylabel("Skor")
    plt.ylim(0, 1.0)
    plt.title("Model Karşılaştırması — Test Seti (Mimari Ablasyon)", fontsize=13)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    png_yol = ortak.SONUCLAR_DIR / "model_karsilastirma.png"
    plt.savefig(png_yol, dpi=300)
    plt.close()
    print(f"[+] Grafik kaydedildi: {png_yol}")


if __name__ == "__main__":
    main()
