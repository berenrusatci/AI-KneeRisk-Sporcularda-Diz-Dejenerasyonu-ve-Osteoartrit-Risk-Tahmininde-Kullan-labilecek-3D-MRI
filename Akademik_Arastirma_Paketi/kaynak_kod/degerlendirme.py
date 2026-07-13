# -*- coding: utf-8 -*-
"""
degerlendirme.py — Ana Model Değerlendirmesi (Test Seti)
========================================================

`best_model_balanced_crop74.pth` modelini bağımsız test setinde (data/test)
değerlendirir ve akademik metrikleri hem ekrana yazar hem de `sonuclar/`
klasörüne dosya olarak kaydeder:

  - metrikler.json            : tüm sayısal metrikler (accuracy, precision/recall/F1,
                                duyarlılık/özgüllük, ROC-AUC, PR-AUC, karışıklık matrisi)
  - siniflandirma_raporu.txt  : sklearn classification_report metni
  - karmasiklik_matrisi.png   : karışıklık matrisi ısı haritası
  - roc_egrisi.png            : ROC eğrisi + AUC
  - pr_egrisi.png             : Precision-Recall eğrisi + AP

Pozitif sınıf = 'abnormal' (etiket 1, klinik olarak "hasta").
Duyarlılık (sensitivity) = hastaları yakalama oranı, Özgüllük (specificity) =
sağlıklıları doğru eleme oranı.

Kullanım:
    python degerlendirme.py
"""

import os
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")   # başsız (GUI'siz) çizim
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)

import ortak


# ============================================================
# Çıkarım: test setinden y_true, y_pred, y_prob(abnormal) topla
# ============================================================
@torch.no_grad()
def run_inference(model, dataset_dir, device, batch_size=16):
    """Verilen split klasöründe çıkarım yapar.

    Dönüş: (y_true, y_pred, y_prob) — y_prob = P(abnormal) (sınıf 1 olasılığı).
    """
    dataset = ortak.Medical25DDataset(root_dir=dataset_dir, transform=ortak.TEST_TRANSFORMS)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    y_true, y_pred, y_prob = [], [], []
    model.eval()
    for inputs, labels in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)          # [B, 2]
        preds = torch.argmax(probs, dim=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_prob.extend(probs[:, 1].cpu().numpy().tolist())   # P(abnormal)

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


# ============================================================
# Metrik hesaplama
# ============================================================
def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # abnormal recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # normal recall
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0           # abnormal precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    report = classification_report(
        y_true, y_pred, labels=[0, 1],
        target_names=ortak.DISPLAY_NAMES, digits=4, output_dict=True,
    )

    # ROC / PR (pozitif = abnormal)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    metrics = {
        "model": os.path.basename(str(ortak.MODEL_PATH)),
        "test_ornek_sayisi": int(len(y_true)),
        "sinif_dagilimi": {
            "normal": int((y_true == 0).sum()),
            "abnormal": int((y_true == 1).sum()),
        },
        "accuracy": float(acc),
        "duyarlilik_sensitivity": float(sensitivity),
        "ozgulluk_specificity": float(specificity),
        "ppv_precision_abnormal": float(ppv),
        "npv": float(npv),
        "roc_auc": float(roc_auc),
        "pr_auc_average_precision": float(ap),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "karisiklik_matrisi": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "sinif_bazli": {
            ortak.DISPLAY_NAMES[0]: report[ortak.DISPLAY_NAMES[0]],
            ortak.DISPLAY_NAMES[1]: report[ortak.DISPLAY_NAMES[1]],
        },
        "_egri_verileri": {   # çizim için (JSON'a da yazılır, tekrar üretilebilir)
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "precision": prec.tolist(), "recall": rec.tolist(),
        },
    }
    return metrics, cm, (fpr, tpr, roc_auc), (prec, rec, ap)


# ============================================================
# Çizimler
# ============================================================
def plot_confusion(cm, out_path):
    plt.figure(figsize=(7, 6), facecolor="white")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=ortak.DISPLAY_NAMES, yticklabels=ortak.DISPLAY_NAMES,
                annot_kws={"size": 16}, cbar=True)
    plt.title("Test Verisi Karışıklık Matrisi (ConvNeXt-Tiny 2.5D, Dengeli)", fontsize=13)
    plt.xlabel("Modelin Tahmini", fontsize=12)
    plt.ylabel("Gerçek Durum", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_roc(fpr, tpr, roc_auc, out_path):
    plt.figure(figsize=(7, 6), facecolor="white")
    plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Rastgele (0.50)")
    plt.xlim([0, 1]); plt.ylim([0, 1.02])
    plt.xlabel("Yanlış Pozitif Oranı (1 - Özgüllük)", fontsize=12)
    plt.ylabel("Doğru Pozitif Oranı (Duyarlılık)", fontsize=12)
    plt.title("ROC Eğrisi — Abnormal (Hasta) Tespiti", fontsize=13)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_pr(prec, rec, ap, out_path, pos_rate):
    plt.figure(figsize=(7, 6), facecolor="white")
    plt.plot(rec, prec, color="#d62728", lw=2.5, label=f"PR (AP = {ap:.4f})")
    plt.axhline(y=pos_rate, color="gray", lw=1.5, linestyle="--",
                label=f"Taban çizgisi ({pos_rate:.3f})")
    plt.xlim([0, 1]); plt.ylim([0, 1.02])
    plt.xlabel("Duyarlılık (Recall)", fontsize=12)
    plt.ylabel("Kesinlik (Precision)", fontsize=12)
    plt.title("Precision-Recall Eğrisi — Abnormal (Hasta) Tespiti", fontsize=13)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================
# Ana
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Diz MR modeli test değerlendirmesi")
    parser.add_argument("--model", default=str(ortak.MODEL_PATH), help="Model .pth yolu")
    parser.add_argument("--test-dir", default=str(ortak.TEST_DIR), help="Test klasörü")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    ortak.SONUCLAR_DIR.mkdir(parents=True, exist_ok=True)
    device = ortak.get_device()
    print(f"[*] Cihaz: {device}")
    print(f"[*] Model: {args.model}")

    model = ortak.load_model(model_path=args.model, device=device)
    print(f"[*] Test verisi yükleniyor: {args.test_dir}")

    y_true, y_pred, y_prob = run_inference(model, args.test_dir, device, args.batch_size)
    print(f"[*] Değerlendirilen hacim sayısı: {len(y_true)}")

    metrics, cm, roc_data, pr_data = compute_metrics(y_true, y_pred, y_prob)

    # --- Ekran raporu ---
    print("\n" + "=" * 60)
    print("KLİNİK SINIFLANDIRMA RAPORU (Test Seti)")
    print("=" * 60)
    report_txt = classification_report(y_true, y_pred, labels=[0, 1],
                                       target_names=ortak.DISPLAY_NAMES, digits=4)
    print(report_txt)
    print(f"Doğruluk (Accuracy)        : {metrics['accuracy']*100:.2f}%")
    print(f"Duyarlılık (Sensitivity)   : {metrics['duyarlilik_sensitivity']*100:.2f}%")
    print(f"Özgüllük (Specificity)     : {metrics['ozgulluk_specificity']*100:.2f}%")
    print(f"ROC-AUC                    : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC (Average Precision) : {metrics['pr_auc_average_precision']:.4f}")
    print(f"Karışıklık Matrisi (TN,FP,FN,TP): {metrics['karisiklik_matrisi']}")
    print("=" * 60)

    # --- Dosyalar ---
    with open(ortak.SONUCLAR_DIR / "metrikler.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(ortak.SONUCLAR_DIR / "siniflandirma_raporu.txt", "w", encoding="utf-8") as f:
        f.write("Diz MR Sınıflandırma — Test Seti Raporu\n")
        f.write(f"Model: {metrics['model']}\n")
        f.write("=" * 60 + "\n")
        f.write(report_txt + "\n")
        f.write(f"Accuracy    : {metrics['accuracy']*100:.2f}%\n")
        f.write(f"Sensitivity : {metrics['duyarlilik_sensitivity']*100:.2f}%\n")
        f.write(f"Specificity : {metrics['ozgulluk_specificity']*100:.2f}%\n")
        f.write(f"ROC-AUC     : {metrics['roc_auc']:.4f}\n")
        f.write(f"PR-AUC      : {metrics['pr_auc_average_precision']:.4f}\n")

    fpr, tpr, roc_auc = roc_data
    prec, rec, ap = pr_data
    pos_rate = metrics["sinif_dagilimi"]["abnormal"] / metrics["test_ornek_sayisi"]

    plot_confusion(cm, ortak.SONUCLAR_DIR / "karmasiklik_matrisi.png")
    plot_roc(fpr, tpr, roc_auc, ortak.SONUCLAR_DIR / "roc_egrisi.png")
    plot_pr(prec, rec, ap, ortak.SONUCLAR_DIR / "pr_egrisi.png", pos_rate)

    print(f"\n[+] Sonuçlar kaydedildi: {ortak.SONUCLAR_DIR}")
    print("    - metrikler.json, siniflandirma_raporu.txt")
    print("    - karmasiklik_matrisi.png, roc_egrisi.png, pr_egrisi.png")


if __name__ == "__main__":
    main()
