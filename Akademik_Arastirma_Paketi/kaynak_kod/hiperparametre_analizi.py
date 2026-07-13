# -*- coding: utf-8 -*-
"""
hiperparametre_analizi.py — Hiperparametre Analizi
==================================================

İki bölümden oluşur:

  A) DOKÜMANTE ANALİZ (her zaman): Nihai modelin eğitiminde kullanılan
     hiperparametreler, değerleri ve gerekçeleri yapılandırılmış biçimde
     raporlanır (kaynak: convext-tiny-focalloss-centercrop-001.py).

  B) AMPİRİK MİNİ-TARAMA (--tara): ConvNeXt-Tiny omurgası DONDURULUP yalnızca
     sınıflandırma başı, küçük bir alt küme üzerinde birkaç epoch eğitilerek
     bazı hiperparametrelerin (öğrenme oranı, kayıp fonksiyonu, sınıf ağırlığı)
     doğrulama başarısına etkisi ÖLÇÜLÜR. Bu, tam yeniden eğitim maliyetine
     girmeden hiperparametre duyarlılığına ampirik kanıt sağlar (linear-probe).

Çıktı:
  - sonuclar/hiperparametre_analizi.json
  - sonuclar/hiperparametre_tarama.png     (yalnızca --tara ile)

Kullanım:
    python hiperparametre_analizi.py            # sadece dokümante analiz
    python hiperparametre_analizi.py --tara     # + ampirik mini-tarama
"""

import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ortak


# ============================================================
# A) DOKÜMANTE HİPERPARAMETRELER (nihai model)
# ============================================================
DOKUMANTE = {
    "mimari": "ConvNeXt-Tiny (ImageNet ön-eğitimli)",
    "giris": "2.5D — orta 3 kesit, 3 kanal, 224x224",
    "batch_size": {"deger": 16, "gerekce": "8-12 GB VRAM ile AMP altında dengeli bellek/hız."},
    "kayip_fonksiyonu": {"deger": "Focal Loss (γ=2.0)",
                          "gerekce": "Kolay örneklerin katkısını azaltıp zor/az örneklenen sınıfa odaklanır; sınıf dengesizliğinde CrossEntropy'den üstündür."},
    "sinif_agirligi": {"deger": "[1.0, 1.25]",
                        "gerekce": "Abnormal (azınlık) sınıfına ılımlı ağırlık; aşırı ağırlık (ör. 1.6) yanlış-pozitifleri (evham) artırdığından 1.25'e düşürülmüştür."},
    "transfer_ogrenme": {"deger": "2 fazlı",
                          "faz1": "10 epoch — yalnızca sınıflandırma başı (backbone donuk), AdamW lr=1e-3",
                          "faz2": "≤40 epoch — tam ince ayar (backbone açık), AdamW lr=5e-5"},
    "optimizer": {"deger": "AdamW", "weight_decay": 1e-2},
    "lr_scheduler": {"deger": "CosineAnnealingLR", "eta_min": 1e-7, "kapsam": "Faz 2"},
    "erken_durdurma": {"deger": "patience=8 (val loss)", "gerekce": "Overfitting başlamadan en iyi checkpoint korunur."},
    "duzenlilestirme": {"dropout": 0.5,
                        "augmentation": "RandomHorizontalFlip, RandomAffine(±10°, ötele/ölçekle), ColorJitter, RandomErasing"},
    "on_isleme": "CenterCrop(280) -> Resize(224) -> Normalize(ImageNet)",
    "amp": "Otomatik Karışık Hassasiyet (float16) — hız ve VRAM tasarrufu",
}


# ============================================================
# B) AMPİRİK MİNİ-TARAMA yardımcıları
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def ozellik_cikarici(device):
    """ConvNeXt-Tiny gövdesinden 768-boyutlu havuzlanmış özellik üreten donuk çıkarıcı.

    Akış: features -> avgpool -> LayerNorm2d -> Flatten  =>  [B, 768]
    (Nihai sınıflandırma Linear katmanı hariç; onun yerine lineer baş eğitilir.)
    """
    base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT).to(device)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False
    ln = base.classifier[0]      # LayerNorm2d(768)
    flat = base.classifier[1]    # Flatten

    @torch.no_grad()
    def cikar(x):
        z = base.features(x)
        z = base.avgpool(z)
        z = flat(ln(z))
        return z
    return cikar


@torch.no_grad()
def ozellikleri_cek(split_dir, cikar, device, batch_size=32):
    """Bir split'teki TÜM örnekler için (Z[N,768], y[N]) özelliklerini bir kez çıkarır."""
    ds = ortak.Medical25DDataset(root_dir=split_dir, transform=ortak.TEST_TRANSFORMS)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    Zs, ys = [], []
    for x, y in loader:
        Zs.append(cikar(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(Zs), torch.cat(ys)


def egit_lineer(Ztr, ytr, Zva, yva, cfg, device, epochs=60, seed=42):
    """Önbelleğe alınmış özellikler üzerinde Linear(768,2) başını eğitir; en iyi val acc döndürür."""
    torch.manual_seed(seed)
    Ztr, ytr = Ztr.to(device), ytr.to(device)
    Zva, yva = Zva.to(device), yva.to(device)

    head = nn.Sequential(nn.Dropout(0.5), nn.Linear(Ztr.shape[1], 2)).to(device)
    w = torch.tensor(cfg["class_weight"], dtype=torch.float32).to(device) if cfg["class_weight"] else None
    criterion = FocalLoss(weight=w, gamma=2.0) if cfg["loss"] == "focal" else nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["lr"], weight_decay=1e-2)

    n = Ztr.shape[0]
    bs = 64
    en_iyi = 0.0
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            optimizer.zero_grad()
            loss = criterion(head(Ztr[idx]), ytr[idx])
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            acc = (head(Zva).argmax(1) == yva).float().mean().item()
        en_iyi = max(en_iyi, acc)
    return en_iyi


def ampirik_tarama(device, epochs=60):
    print(f"[*] Ampirik hiperparametre taraması (özellik-önbellekli linear-probe)")
    print(f"    TÜM train/val seti, donuk ConvNeXt omurgası, baş {epochs} epoch\n")
    cikar = ozellik_cikarici(device)
    print("[*] Özellikler çıkarılıyor (train)...")
    Ztr, ytr = ozellikleri_cek(ortak.TRAIN_DIR, cikar, device)
    print("[*] Özellikler çıkarılıyor (val)...")
    Zva, yva = ozellikleri_cek(ortak.VAL_DIR, cikar, device)
    print(f"    train={tuple(Ztr.shape)}  val={tuple(Zva.shape)}\n")

    # Kontrollü tek-değişkenli konfigürasyonlar
    taban = {"lr": 1e-3, "loss": "focal", "class_weight": [1.0, 1.25]}
    configs = [
        {"grup": "Öğrenme Oranı", "etiket": "lr=1e-2", **{**taban, "lr": 1e-2}},
        {"grup": "Öğrenme Oranı", "etiket": "lr=1e-3 (seçilen)", **{**taban, "lr": 1e-3}},
        {"grup": "Öğrenme Oranı", "etiket": "lr=1e-4", **{**taban, "lr": 1e-4}},
        {"grup": "Kayıp Fonksiyonu", "etiket": "CrossEntropy", **{**taban, "loss": "ce"}},
        {"grup": "Kayıp Fonksiyonu", "etiket": "Focal (seçilen)", **{**taban, "loss": "focal"}},
        {"grup": "Sınıf Ağırlığı", "etiket": "[1.0, 1.0]", **{**taban, "class_weight": [1.0, 1.0]}},
        {"grup": "Sınıf Ağırlığı", "etiket": "[1.0, 1.25] (seçilen)", **{**taban, "class_weight": [1.0, 1.25]}},
        {"grup": "Sınıf Ağırlığı", "etiket": "[1.0, 1.6]", **{**taban, "class_weight": [1.0, 1.6]}},
    ]

    sonuclar = []
    for c in configs:
        acc = egit_lineer(Ztr, ytr, Zva, yva, c, device, epochs)
        sonuclar.append({"grup": c["grup"], "etiket": c["etiket"], "val_acc": acc})
        print(f"[+] {c['grup']:16s} | {c['etiket']:22s} -> En iyi Val Acc = {acc*100:.2f}%")
    return sonuclar


def tarama_grafigi(sonuclar, out_path):
    gruplar = ["Öğrenme Oranı", "Kayıp Fonksiyonu", "Sınıf Ağırlığı"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
    for ax, g in zip(axes, gruplar):
        alt = [s for s in sonuclar if s["grup"] == g]
        etk = [s["etiket"] for s in alt]
        vals = [s["val_acc"] * 100 for s in alt]
        renkler = ["#ff7f0e" if "seçilen" in e else "#1f77b4" for e in etk]
        ax.bar(range(len(etk)), vals, color=renkler)
        ax.set_xticks(range(len(etk)))
        ax.set_xticklabels(etk, rotation=20, ha="right", fontsize=8)
        ax.set_title(g, fontsize=12)
        ax.set_ylabel("Doğrulama Doğruluğu (%)")
        ax.set_ylim(40, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    fig.suptitle("Hiperparametre Duyarlılığı — Ampirik Mini-Tarama (linear-probe)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Hiperparametre analizi")
    parser.add_argument("--tara", action="store_true", help="Ampirik hiperparametre taraması da çalıştır")
    parser.add_argument("--epochs", type=int, default=60, help="Lineer baş eğitim epoch sayısı")
    args = parser.parse_args()

    ortak.SONUCLAR_DIR.mkdir(parents=True, exist_ok=True)
    cikti = {"dokumante_hiperparametreler": DOKUMANTE}

    print("=" * 60)
    print("HİPERPARAMETRE ANALİZİ — Dokümante Konfigürasyon")
    print("=" * 60)
    for k, v in DOKUMANTE.items():
        print(f"  {k}: {v if not isinstance(v, dict) else v.get('deger', v)}")

    if args.tara:
        device = ortak.get_device()
        print(f"\n[*] Cihaz: {device}")
        sonuclar = ampirik_tarama(device, args.epochs)
        cikti["ampirik_tarama"] = {
            "ayarlar": {"veri": "tüm train/val seti", "epochs": args.epochs,
                        "yontem": "özellik-önbellekli linear-probe (donuk ConvNeXt omurgası)"},
            "sonuclar": sonuclar,
        }
        tarama_grafigi(sonuclar, ortak.SONUCLAR_DIR / "hiperparametre_tarama.png")
        print(f"\n[+] Grafik: hiperparametre_tarama.png")

    with open(ortak.SONUCLAR_DIR / "hiperparametre_analizi.json", "w", encoding="utf-8") as f:
        json.dump(cikti, f, indent=2, ensure_ascii=False)
    print(f"[+] JSON: hiperparametre_analizi.json")


if __name__ == "__main__":
    main()
