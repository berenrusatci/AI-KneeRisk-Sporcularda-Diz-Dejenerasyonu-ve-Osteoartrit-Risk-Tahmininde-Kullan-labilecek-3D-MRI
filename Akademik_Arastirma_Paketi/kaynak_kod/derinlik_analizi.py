# -*- coding: utf-8 -*-
"""
derinlik_analizi.py — Tüm Derinliği Kullanma: Kayan Pencere + Birleştirme Ablasyonu
===================================================================================

Ana model (`best_model_balanced_crop74.pth`) her hacimden yalnızca ORTADAKİ 3 kesiti
kullanır (2.5D). Bu script, YENİDEN EĞİTİM OLMADAN, mevcut ağırlıkla hacmin TAMAMINI
3'erli kayan pencerelerle tarayıp pencere-başı P(abnormal) skorlarını birleştirerek
(mean / max / top-k) tek hacim kararı üretir (MRNet tarzı çıkarım-zamanı birleştirme)
ve bunu mevcut orta-3 TEMELİYLE karşılaştırır.

AKADEMİK YÖNTEM (test sızıntısını önlemek için):
  - Birincil karşılaştırma EŞİKTEN BAĞIMSIZ metriklerledir: ROC-AUC, PR-AUC.
  - Sabit 0.50 eşiğinde accuracy/duyarlılık/özgüllük da raporlanır (argmax referansı).
  - Ek olarak, her skorlama için eşik YALNIZCA val'de seçilip (Youden-J) test metrikleri
    ayrı bir kolonda verilir — sızıntısız accuracy karşılaştırması.

Model orta kesitlerle eğitildiğinden uç kesitler dağılım-dışıdır; `max` birleştirmenin
özgüllüğü düşürmesi (yanlış alarm↑), `mean`'in bulguyu sulandırması beklenir. Hangisinin
gerçekten kazandığı ancak burada ölçülür.

Çıktı:
  - sonuclar/derinlik_analizi.json   (tam sonuçlar: yöntem başına test+val metrikleri)
  - sonuclar/derinlik_analizi.csv    (özet tablo)
  - sonuclar/derinlik_analizi.png    (yöntem × metrik çubuk grafiği, temel referanslı)

Kullanım:
    python derinlik_analizi.py                       # step=1, k=[3,5]
    python derinlik_analizi.py --step 3 --k 3 5 10   # daha hızlı, farklı k
"""

import argparse
import json
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

import ortak
from degerlendirme import run_inference   # orta-3 temeli için (aynı çıkarım yolu)
from esik_analizi import metrik_hesapla   # tek eşikte accuracy/sens/spec/... (yeniden kullanım)


# ------------------------------------------------------------
# Kayan pencere: bir split'teki tüm hacimler için pencere skorları
# ------------------------------------------------------------
@torch.no_grad()
def kayan_pencere_skorlari(model, split_dir, device, window=3, step=1, batch_size=64):
    """Split'teki her hacim için pencere-başı P(abnormal) skorlarını toplar.

    Yeniden kullanım: hacim listesi Medical25DDataset.samples (≥3 PNG filtreli).
    Dönüş: (y_true: np.ndarray, skor_listesi: list[list[float]]) — hizalı.
    """
    ds = ortak.Medical25DDataset(root_dir=split_dir)   # yalnızca .samples kullanılır
    y_true, tum_skorlar = [], []
    t0 = time.time()
    for i, (sample_path, label) in enumerate(ds.samples, 1):
        skorlar, _ = ortak.sliding_window_scores(
            sample_path, model=model, device=device,
            window=window, step=step, batch_size=batch_size)
        y_true.append(label)
        tum_skorlar.append(skorlar)
        if i % 50 == 0 or i == len(ds.samples):
            print(f"    {i}/{len(ds.samples)} hacim  ({time.time()-t0:.0f} sn)")
    return np.array(y_true), tum_skorlar


def birlestir(tum_skorlar, method, k):
    """Her hacmin pencere skorlarını tek skora indirger -> np.ndarray [n_hacim]."""
    return np.array([ortak.aggregate_scores(s, method=method, k=k) for s in tum_skorlar])


# ------------------------------------------------------------
# Metrik paketi (eşiksiz AUC'lar + sabit/val eşik metrikleri)
# ------------------------------------------------------------
def youden_esigi(y_true, y_prob):
    """val'de duyarlılık+özgüllük dengesini (Youden-J) en iyileyen eşik."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(np.clip(thr[int(np.argmax(j))], 0.001, 0.999))


def metrik_paketi(ad, yt, pt, yv=None, pv=None):
    """Bir skorlama yöntemi için tam metrik sözlüğü.

    yt/pt : test etiket/olasılık ; yv/pv : val (val-seçili eşik için, opsiyonel).
    """
    roc = float(roc_auc_score(yt, pt))
    pr = float(average_precision_score(yt, pt))
    m050 = metrik_hesapla(yt, pt, 0.50)          # sabit 0.50 (argmax referansı)

    kayit = {
        "ad": ad,
        "roc_auc": roc,
        "pr_auc": pr,
        "esik_0.50": m050,
    }
    if yv is not None and pv is not None:
        t_val = youden_esigi(yv, pv)             # eşik SADECE val'de seçilir
        kayit["val_youden_esik"] = t_val
        kayit["esik_val_youden"] = metrik_hesapla(yt, pt, t_val)
    return kayit


def satir(kayit):
    """Özet tablo satırı (test metrikleri)."""
    m = kayit["esik_0.50"]
    mv = kayit.get("esik_val_youden", m)
    return {
        "Yöntem": kayit["ad"],
        "ROC-AUC": kayit["roc_auc"],
        "PR-AUC": kayit["pr_auc"],
        "Doğruluk@0.50": m["accuracy"],
        "Duyarlılık@0.50": m["duyarlilik"],
        "Özgüllük@0.50": m["ozgulluk"],
        "FN@0.50": m["FN"],
        "FP@0.50": m["FP"],
        "Doğruluk@val-eşik": mv["accuracy"],
    }


# ------------------------------------------------------------
# Grafik
# ------------------------------------------------------------
def grafik_ciz(kayitlar, temel_ad, out_path):
    adlar = [k["ad"] for k in kayitlar]
    roc = [k["roc_auc"] for k in kayitlar]
    acc = [k["esik_0.50"]["accuracy"] for k in kayitlar]
    sens = [k["esik_0.50"]["duyarlilik"] for k in kayitlar]
    spec = [k["esik_0.50"]["ozgulluk"] for k in kayitlar]

    temel = next(k for k in kayitlar if k["ad"] == temel_ad)

    x = np.arange(len(adlar))
    w = 0.2
    fig, ax = plt.subplots(figsize=(max(9, 1.7 * len(adlar)), 6), facecolor="white")
    ax.bar(x - 1.5 * w, roc, w, label="ROC-AUC", color="#1f77b4")
    ax.bar(x - 0.5 * w, acc, w, label="Doğruluk@0.50", color="#2ca02c")
    ax.bar(x + 0.5 * w, sens, w, label="Duyarlılık@0.50", color="#d62728")
    ax.bar(x + 1.5 * w, spec, w, label="Özgüllük@0.50", color="#ff7f0e")

    # Temel referans çizgileri
    ax.axhline(temel["roc_auc"], color="#1f77b4", ls="--", lw=1, alpha=0.7)
    ax.axhline(temel["esik_0.50"]["accuracy"], color="#2ca02c", ls="--", lw=1, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(adlar, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Skor")
    ax.set_title("Tüm Derinlik (Kayan Pencere) vs Orta-3 Temeli — Test Seti\n"
                 "(kesikli çizgiler = orta-3 temeli ROC-AUC / Doğruluk)")
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ------------------------------------------------------------
# Ana
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Kayan pencere + birleştirme ablasyonu")
    ap.add_argument("--step", type=int, default=1, help="Kayan pencere adımı (varsayılan 1)")
    ap.add_argument("--k", type=int, nargs="+", default=[3, 5], help="top-k için k değerleri")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-val", action="store_true",
                    help="val-seçili eşik kolonunu atla (yalnızca test)")
    args = ap.parse_args()

    ortak.SONUCLAR_DIR.mkdir(parents=True, exist_ok=True)
    device = ortak.get_device()
    print(f"[*] Cihaz: {device}")
    print(f"[*] Model: {ortak.MODEL_PATH.name}")
    model = ortak.load_model(device=device)

    # --- 1) Orta-3 TEMEL (test) — mevcut çıkarım yolu ---
    print("\n[1] Orta-3 temel (test) çıkarımı...")
    yt_base, _, pt_base = run_inference(model, ortak.TEST_DIR, device)
    yv_base = pv_base = None
    if not args.no_val:
        print("[1b] Orta-3 temel (val) çıkarımı (val-seçili eşik için)...")
        yv_base, _, pv_base = run_inference(model, ortak.VAL_DIR, device)

    # --- 2) Kayan pencere skorları (test + val) ---
    print(f"\n[2] Kayan pencere skorları (test, step={args.step})...")
    yt, test_skorlar = kayan_pencere_skorlari(model, ortak.TEST_DIR, device,
                                              step=args.step, batch_size=args.batch_size)
    assert np.array_equal(yt, yt_base), "Test etiket sırası temelle uyuşmuyor!"
    yv = val_skorlar = None
    if not args.no_val:
        print(f"[2b] Kayan pencere skorları (val, step={args.step})...")
        yv, val_skorlar = kayan_pencere_skorlari(model, ortak.VAL_DIR, device,
                                                 step=args.step, batch_size=args.batch_size)

    # --- 3) Metrikler: temel + her birleştirme ---
    TEMEL_AD = "Orta-3 (temel)"
    kayitlar = [metrik_paketi(TEMEL_AD, yt_base, pt_base, yv_base, pv_base)]

    yontemler = [("mean", None), ("max", None)] + [("topk", k) for k in args.k]
    for method, k in yontemler:
        ad = "Kayan: mean" if method == "mean" else \
             "Kayan: max" if method == "max" else f"Kayan: top-{k} ort."
        pt = birlestir(test_skorlar, method, k or 0)
        pv = birlestir(val_skorlar, method, k or 0) if val_skorlar is not None else None
        kayitlar.append(metrik_paketi(ad, yt, pt, yv, pv))

    # --- 4) Tablo ---
    df = pd.DataFrame([satir(k) for k in kayitlar])
    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print(f"TÜM DERİNLİK KARŞILAŞTIRMASI (TEST seti, n={len(yt)}, step={args.step})")
    print("=" * 100)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("=" * 100)

    # --- 5) Kazanan özeti (temele göre fark) ---
    temel = kayitlar[0]
    print(f"\n[ÖZET] Orta-3 temeli: ROC-AUC {temel['roc_auc']:.4f}, "
          f"Doğruluk@0.50 {temel['esik_0.50']['accuracy']*100:.1f}%")
    en_iyi_roc = temel
    for k in kayitlar[1:]:
        d_roc = k["roc_auc"] - temel["roc_auc"]
        d_acc = k["esik_0.50"]["accuracy"] - temel["esik_0.50"]["accuracy"]
        isaret = "▲" if d_roc > 0 else "▼"
        print(f"  {k['ad']:<22} ROC-AUC {k['roc_auc']:.4f} ({isaret}{d_roc:+.4f}), "
              f"Doğruluk {k['esik_0.50']['accuracy']*100:5.1f}% ({d_acc*100:+.1f} puan)")
        if k["roc_auc"] > en_iyi_roc["roc_auc"]:
            en_iyi_roc = k

    if en_iyi_roc["ad"] == TEMEL_AD:
        print(f"\n[KARAR] Hiçbir kayan pencere yöntemi temeli ROC-AUC'de geçemedi -> "
              f"entegrasyon ÖNERİLMEZ (ablasyon sonucu).")
    else:
        print(f"\n[KARAR] En yüksek ROC-AUC: '{en_iyi_roc['ad']}' "
              f"({en_iyi_roc['roc_auc']:.4f}, temele göre "
              f"{en_iyi_roc['roc_auc']-temel['roc_auc']:+.4f}) -> "
              f"arayüz/exe entegrasyonu için aday.")

    # --- 6) Dosyalar ---
    with open(ortak.SONUCLAR_DIR / "derinlik_analizi.json", "w", encoding="utf-8") as f:
        json.dump({
            "aciklama": "Kayan pencere + birleştirme ablasyonu. AUC'lar eşiksiz; "
                        "0.50 sabit eşik argmax referansı; val-youden eşik sızıntısız.",
            "step": args.step, "k_degerleri": args.k,
            "test_ornek_sayisi": int(len(yt)),
            "yontemler": kayitlar,
        }, f, indent=2, ensure_ascii=False)
    df.to_csv(ortak.SONUCLAR_DIR / "derinlik_analizi.csv", index=False,
              encoding="utf-8-sig", float_format="%.4f")
    grafik_ciz(kayitlar, TEMEL_AD, ortak.SONUCLAR_DIR / "derinlik_analizi.png")

    print(f"\n[+] Kaydedildi: derinlik_analizi.json / .csv / .png  ->  {ortak.SONUCLAR_DIR}")


if __name__ == "__main__":
    main()
