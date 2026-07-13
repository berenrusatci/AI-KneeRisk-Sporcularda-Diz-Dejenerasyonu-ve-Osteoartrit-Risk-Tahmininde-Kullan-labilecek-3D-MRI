# -*- coding: utf-8 -*-
"""
gap_overfitting_analizi.py — GAP (Genelleme Boşluğu) ve Overfitting Analizi
===========================================================================

Ana model (best_model_balanced_crop74.pth) AYNI deterministik ön işlemeyle
train / val / test setlerinin ÜÇÜNDE de değerlendirilir. Amaç, modelin eğitim
verisini ezberleyip ezberlemediğini (overfitting) ölçmek ve genelleme boşluğunu
(GAP = eğitim başarısı − test başarısı) sayısallaştırmaktır.

Yorum kılavuzu:
  - GAP ≈ 0            : ezber yok, iyi genelleme
  - GAP küçük-orta     : hafif overfitting (kabul edilebilir)
  - GAP büyük (>%15-20): belirgin overfitting

Not: Eğitim seti burada AUGMENTATION'SIZ (deterministik) değerlendirilir; bu,
"temiz" eğitim başarısını verir ve ezber ölçümü için doğru referanstır.

Ayrıca mevcut öğrenme eğrileri görseli (ogrenme_egrileri_balanced.png) pakete
kopyalanır.

Çıktı:
  - sonuclar/gap_overfitting.json
  - sonuclar/gap_overfitting.png
  - sonuclar/ogrenme_egrileri.png  (varsa kopya)

Kullanım:
    python gap_overfitting_analizi.py
"""

import json
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import ortak
from degerlendirme import run_inference


def split_metrikleri(model, split_dir, device):
    yt, yp, ypr = run_inference(model, split_dir, device)
    return {
        "n": int(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro")),
        "roc_auc": float(roc_auc_score(yt, ypr)),
    }


def main():
    ortak.SONUCLAR_DIR.mkdir(parents=True, exist_ok=True)
    device = ortak.get_device()
    print(f"[*] Cihaz: {device}")
    print(f"[*] Model: {ortak.MODEL_PATH.name}\n")

    model = ortak.load_model(device=device)

    splitler = {"Eğitim (train)": ortak.TRAIN_DIR,
                "Doğrulama (val)": ortak.VAL_DIR,
                "Test": ortak.TEST_DIR}

    sonuc = {}
    for ad, yol in splitler.items():
        print(f"[*] {ad} değerlendiriliyor...")
        sonuc[ad] = split_metrikleri(model, yol, device)
        s = sonuc[ad]
        print(f"    n={s['n']:5d} | Acc={s['accuracy']*100:5.2f}%  "
              f"F1={s['macro_f1']:.3f}  AUC={s['roc_auc']:.3f}")

    tr = sonuc["Eğitim (train)"]
    va = sonuc["Doğrulama (val)"]
    te = sonuc["Test"]

    gap = {
        "gap_accuracy_train_test": float(tr["accuracy"] - te["accuracy"]),
        "gap_accuracy_train_val": float(tr["accuracy"] - va["accuracy"]),
        "gap_auc_train_test": float(tr["roc_auc"] - te["roc_auc"]),
        "gap_f1_train_test": float(tr["macro_f1"] - te["macro_f1"]),
    }

    ga = gap["gap_accuracy_train_test"] * 100
    if ga < 5:
        yorum = "İhmal edilebilir GAP: ezber yok, çok iyi genelleme."
    elif ga < 15:
        yorum = "Küçük-orta GAP: hafif overfitting, akademik olarak kabul edilebilir."
    else:
        yorum = "Büyük GAP: belirgin overfitting; düzenlileştirme/veri artırımı önerilir."

    cikti = {
        "model": ortak.MODEL_PATH.name,
        "split_metrikleri": sonuc,
        "gap": gap,
        "yorum": yorum,
    }
    with open(ortak.SONUCLAR_DIR / "gap_overfitting.json", "w", encoding="utf-8") as f:
        json.dump(cikti, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("GAP (GENELLEME BOŞLUĞU) ANALİZİ")
    print("=" * 60)
    print(f"Eğitim Doğruluğu : {tr['accuracy']*100:.2f}%")
    print(f"Test Doğruluğu   : {te['accuracy']*100:.2f}%")
    print(f"GAP (train-test) : {ga:.2f} puan")
    print(f"Yorum            : {yorum}")
    print("=" * 60)

    # --- Grafik: split bazlı Acc/F1/AUC ---
    adlar = list(sonuc.keys())
    accs = [sonuc[a]["accuracy"] for a in adlar]
    f1s = [sonuc[a]["macro_f1"] for a in adlar]
    aucs = [sonuc[a]["roc_auc"] for a in adlar]

    x = np.arange(len(adlar)); w = 0.25
    plt.figure(figsize=(10, 6), facecolor="white")
    plt.bar(x - w, accs, w, label="Doğruluk", color="#1f77b4")
    plt.bar(x, f1s, w, label="Makro-F1", color="#2ca02c")
    plt.bar(x + w, aucs, w, label="ROC-AUC", color="#ff7f0e")
    for i, v in enumerate(accs):
        plt.text(x[i] - w, v + 0.01, f"{v*100:.1f}", ha="center", fontsize=9)
    plt.xticks(x, adlar)
    plt.ylim(0, 1.05)
    plt.ylabel("Skor")
    plt.title(f"Train / Val / Test Başarısı ve Genelleme Boşluğu\n"
              f"(GAP train-test = {ga:.1f} puan)", fontsize=12)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(ortak.SONUCLAR_DIR / "gap_overfitting.png", dpi=300)
    plt.close()
    print(f"[+] Kaydedildi: gap_overfitting.json, gap_overfitting.png")

    # --- Öğrenme eğrilerini pakete kopyala ---
    kaynak_egri = ortak.PROJE_KOK / "ogrenme_egrileri_balanced.png"
    if kaynak_egri.exists():
        shutil.copy2(kaynak_egri, ortak.SONUCLAR_DIR / "ogrenme_egrileri.png")
        print(f"[+] Öğrenme eğrileri kopyalandı: sonuclar/ogrenme_egrileri.png")
    else:
        print(f"[-] Öğrenme eğrileri görseli bulunamadı: {kaynak_egri}")


if __name__ == "__main__":
    main()
