# -*- coding: utf-8 -*-
"""
esik_analizi.py — Karar Eşiği (Threshold) Ayarı: Duyarlılığı Artırma
====================================================================

Model, varsayılan olarak argmax (P(abnormal) >= 0.50) ile karar verir. Tarama
senaryolarında hastaları kaçırmamak (yüksek DUYARLILIK) önceliklidir. Bu script,
karar eşiğini düşürerek duyarlılığı artırır ve özgüllük ile arasındaki ödünleşimi
raporlar.

AKADEMİK YÖNTEM (test sızıntısını önlemek için):
  - Eşikler YALNIZCA doğrulama (val) setinde seçilir.
  - Seçilen eşiklerle nihai metrikler bağımsız TEST setinde raporlanır.

Değerlendirilen eşikler:
  - Standart (0.50)              : argmax (referans)
  - Youden-J (val)              : duyarlılık+özgüllük dengesini maksimize eder
  - Duyarlılık ≥ 0.85 (val)     : hedef duyarlılık (ÖNERİLEN)
  - Duyarlılık ≥ 0.90 (val)     : daha agresif tarama

Çıktı:
  - sonuclar/esik_analizi.json     (tam sonuçlar)
  - sonuclar/esik_analizi.csv      (karşılaştırma tablosu)
  - sonuclar/esik_analizi.png      (duyarlılık/özgüllük vs eşik)
  - sonuclar/esik_ayari.json       (ÖNERİLEN eşik — UI/CLI bunu okur)

Kullanım:
    python esik_analizi.py
"""

import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix

import ortak
from degerlendirme import run_inference

HEDEF_DUYARLILIK_ONERILEN = 0.85


def metrik_hesapla(y_true, y_prob, t):
    y_pred = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) else 0.0        # abnormal precision (PPV)
    f1_abn = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    prec0 = tn / (tn + fn) if (tn + fn) else 0.0
    rec0 = spec
    f1_0 = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) else 0.0
    return {
        "esik": float(t), "accuracy": float(acc),
        "duyarlilik": float(sens), "ozgulluk": float(spec),
        "ppv_precision_abnormal": float(prec),
        "f1_abnormal": float(f1_abn), "macro_f1": float((f1_abn + f1_0) / 2),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def youden_esigi(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    J = tpr - fpr
    t = thr[int(np.argmax(J))]
    return float(np.clip(t, 0.001, 0.999))


def hedef_duyarlilik_esigi(y_true, y_prob, hedef):
    """val'de duyarlılık >= hedef sağlayan EN YÜKSEK eşik (özgüllüğü en çok koruyan)."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    uygun = thr[(tpr >= hedef) & np.isfinite(thr)]
    if len(uygun) == 0:
        return 0.001
    return float(np.clip(uygun.max(), 0.001, 0.999))


def main():
    ortak.SONUCLAR_DIR.mkdir(parents=True, exist_ok=True)
    device = ortak.get_device()
    model = ortak.load_model(device=device)
    print(f"[*] Cihaz: {device}\n[*] Model: {ortak.MODEL_PATH.name}\n")

    print("[*] Doğrulama (val) çıkarımı...")
    yv, _, pv = run_inference(model, ortak.VAL_DIR, device)
    print("[*] Test çıkarımı...")
    yt, _, pt = run_inference(model, ortak.TEST_DIR, device)

    # Eşikler val'de seçilir
    t_std = 0.50
    t_youden = youden_esigi(yv, pv)
    t_s85 = hedef_duyarlilik_esigi(yv, pv, 0.85)
    t_s90 = hedef_duyarlilik_esigi(yv, pv, 0.90)

    adaylar = [
        ("Standart (0.50)", t_std),
        ("Youden-J (val)", t_youden),
        ("Duyarlılık ≥ 0.85 (val) [ÖNERİLEN]", t_s85),
        ("Duyarlılık ≥ 0.90 (val)", t_s90),
    ]

    tablo = []
    tam = {}
    for ad, t in adaylar:
        test_m = metrik_hesapla(yt, pt, t)
        val_m = metrik_hesapla(yv, pv, t)
        tam[ad] = {"esik": t, "test": test_m, "val": val_m}
        tablo.append({
            "Senaryo": ad, "Eşik": round(t, 3),
            "Doğruluk": test_m["accuracy"], "Duyarlılık": test_m["duyarlilik"],
            "Özgüllük": test_m["ozgulluk"], "PPV": test_m["ppv_precision_abnormal"],
            "Makro-F1": test_m["macro_f1"], "FN": test_m["FN"], "FP": test_m["FP"],
        })

    df = pd.DataFrame(tablo)
    print("\n" + "=" * 78)
    print("KARAR EŞİĞİ KARŞILAŞTIRMASI (metrikler TEST setinde)")
    print("=" * 78)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("=" * 78)

    std_m = tam["Standart (0.50)"]["test"]
    one_m = tam["Duyarlılık ≥ 0.85 (val) [ÖNERİLEN]"]["test"]
    print(f"\n[ÖZET] Standart -> önerilen eşik ({t_s85:.3f}):")
    print(f"  Duyarlılık : %{std_m['duyarlilik']*100:.1f} -> %{one_m['duyarlilik']*100:.1f}  "
          f"(kaçan hasta FN: {std_m['FN']} -> {one_m['FN']})")
    print(f"  Özgüllük   : %{std_m['ozgulluk']*100:.1f} -> %{one_m['ozgulluk']*100:.1f}  "
          f"(yanlış alarm FP: {std_m['FP']} -> {one_m['FP']})")

    # --- Dosyalar ---
    df.to_csv(ortak.SONUCLAR_DIR / "esik_analizi.csv", index=False,
              encoding="utf-8-sig", float_format="%.4f")
    with open(ortak.SONUCLAR_DIR / "esik_analizi.json", "w", encoding="utf-8") as f:
        json.dump({"aciklama": "Eşikler val'de seçildi, metrikler test'te raporlandı.",
                   "adaylar": tam}, f, indent=2, ensure_ascii=False)

    # ÖNERİLEN eşiği UI/CLI için kaydet
    with open(ortak.ESIK_AYAR_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "onerilen_esik": t_s85,
            "kriter": "val duyarlılık >= 0.85",
            "test_metrikleri": one_m,
            "standart_esik": 0.50,
            "standart_test_metrikleri": std_m,
        }, f, indent=2, ensure_ascii=False)

    # --- Grafik: test duyarlılık/özgüllük vs eşik ---
    grid = np.linspace(0.02, 0.98, 97)
    sens = [metrik_hesapla(yt, pt, t)["duyarlilik"] for t in grid]
    spec = [metrik_hesapla(yt, pt, t)["ozgulluk"] for t in grid]
    acc = [metrik_hesapla(yt, pt, t)["accuracy"] for t in grid]

    plt.figure(figsize=(10, 6), facecolor="white")
    plt.plot(grid, sens, label="Duyarlılık (Sensitivity)", color="#d62728", lw=2.5)
    plt.plot(grid, spec, label="Özgüllük (Specificity)", color="#1f77b4", lw=2.5)
    plt.plot(grid, acc, label="Doğruluk", color="#2ca02c", lw=1.5, linestyle="--")
    plt.axvline(0.50, color="gray", linestyle=":", label="Standart eşik (0.50)")
    plt.axvline(t_s85, color="orange", linestyle="-", lw=2,
                label=f"Önerilen eşik ({t_s85:.3f})")
    plt.xlabel("Karar Eşiği — P(abnormal)")
    plt.ylabel("Skor")
    plt.title("Karar Eşiğinin Duyarlılık / Özgüllük Üzerindeki Etkisi (Test Seti)")
    plt.legend(loc="center left", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(ortak.SONUCLAR_DIR / "esik_analizi.png", dpi=300)
    plt.close()

    print(f"\n[+] Kaydedildi: esik_analizi.json/.csv/.png, esik_ayari.json")
    print(f"[+] UI/CLI artık önerilen eşiği ({t_s85:.3f}) 'Yüksek Duyarlılık' modunda kullanabilir.")


if __name__ == "__main__":
    main()
