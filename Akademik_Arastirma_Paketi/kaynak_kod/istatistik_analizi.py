# -*- coding: utf-8 -*-
"""
istatistik_analizi.py — İstatistiksel Anlamlılık ve Güven Aralıkları
====================================================================

Başvuru Yöntem bölümünün istatistiksel gereklerini karşılar:
  (1) Her modelin ROC-AUC ve Doğruluk değeri için **%95 güven aralığı**
      (stratified bootstrap, B tekrar).
  (2) ANA model ile diğer mimariler arasında **eşleştirilmiş (paired)**
      karşılaştırma:
        - ΔROC-AUC için paired bootstrap %95 GA ve iki-yönlü p-değeri,
        - **Wilcoxon signed-rank testi** (Wilcoxon, 1945): örnek-başı "doğru
          sınıfa atanan olasılık" (true-class probability) üzerinde ANA vs diğer.

Tüm modeller AYNI bağımsız test setinde, model_karsilastirma.py'deki kayıt
defteri (MODELLER) ve ön işleme zincirleriyle değerlendirilir (adil, eşleştirilmiş).

Çıktı:
  - sonuclar/istatistik_analizi.json   (GA'lar, p-değerleri, karşılaştırmalar)
  - sonuclar/istatistik_analizi.csv    (model başına AUC/Acc + %95 GA)
  - sonuclar/istatistik_auc_ci.png     (AUC + %95 GA — forest plot)

Kullanım:
    python istatistik_analizi.py [--bootstrap 2000] [--seed 42]
"""

import json
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, accuracy_score

import ortak
from model_karsilastirma import MODELLER, degerlendir


def bootstrap_ci_auc_acc(y_true, y_prob, y_pred, B=2000, seed=42):
    """Tek model için AUC ve Doğruluk %95 güven aralığı (stratified bootstrap)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    aucs, accs = [], []
    for _ in range(B):
        # sınıf oranını koruyan (stratified) yeniden örnekleme
        s_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        s_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        s = np.concatenate([s_pos, s_neg])
        yt, yp, ypr = y_true[s], y_pred[s], y_prob[s]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ypr))
        accs.append(accuracy_score(yt, yp))
    return {
        "auc_ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        "acc_ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
    }


def paired_delta_auc(y_true, prob_a, prob_b, B=2000, seed=42):
    """ANA (a) - diğer (b) ROC-AUC farkı için paired bootstrap %95 GA + p-değeri."""
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    deltas = []
    for _ in range(B):
        s_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        s_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        s = np.concatenate([s_pos, s_neg])
        yt = y_true[s]
        if len(np.unique(yt)) < 2:
            continue
        deltas.append(roc_auc_score(yt, prob_a[s]) - roc_auc_score(yt, prob_b[s]))
    deltas = np.array(deltas)
    # iki-yönlü bootstrap p: farkın 0'ı içerme olasılığı
    p = 2.0 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    p = float(min(1.0, p))
    return {
        "delta_auc_ortalama": float(np.mean(deltas)),
        "delta_auc_ci95": [float(np.percentile(deltas, 2.5)),
                           float(np.percentile(deltas, 97.5))],
        "bootstrap_p": p,
    }


def true_class_prob(y_true, y_prob):
    """Örnek-başı doğru sınıfa atanan olasılık: y=1 -> p, y=0 -> 1-p (yüksek = iyi)."""
    return np.where(y_true == 1, y_prob, 1.0 - y_prob)


def main():
    parser = argparse.ArgumentParser(description="İstatistiksel anlamlılık ve GA analizi")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ortak.SONUCLAR_DIR.mkdir(parents=True, exist_ok=True)
    device = ortak.get_device()
    print(f"[*] Cihaz: {device}\n[*] Test seti: {ortak.TEST_DIR}\n")

    # --- Her model için test çıkarımı (eşleştirilmiş: aynı sıra, aynı y_true) ---
    sonuc = {}   # ad -> {y_true, y_pred, y_prob}
    ref_yt = None
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
        sonuc[cfg["ad"]] = {"yt": yt, "yp": yp, "ypr": ypr}
        if ref_yt is None:
            ref_yt = yt
        print(f"[+] Değerlendirildi: {cfg['ad']}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(sonuc) < 1:
        print("[-] Hiç model değerlendirilemedi.")
        return

    adlar = list(sonuc.keys())
    ana_ad = adlar[0]   # MODELLER'de ilk kayıt ANA modeldir
    y_true = sonuc[ana_ad]["yt"]

    # --- (1) Model başına nokta tahmin + %95 GA ---
    tablo = []
    ga = {}
    for ad in adlar:
        d = sonuc[ad]
        auc = float(roc_auc_score(d["yt"], d["ypr"]))
        acc = float(accuracy_score(d["yt"], d["yp"]))
        ci = bootstrap_ci_auc_acc(d["yt"], d["ypr"], d["yp"], B=args.bootstrap, seed=args.seed)
        ga[ad] = {"auc": auc, "acc": acc, **ci}
        tablo.append({
            "Model": ad,
            "ROC-AUC": auc,
            "AUC %95 GA alt": ci["auc_ci95"][0], "AUC %95 GA üst": ci["auc_ci95"][1],
            "Doğruluk": acc,
            "Acc %95 GA alt": ci["acc_ci95"][0], "Acc %95 GA üst": ci["acc_ci95"][1],
        })

    # --- (2) ANA vs diğerleri: ΔAUC (paired bootstrap) + Wilcoxon signed-rank ---
    karsilastirmalar = {}
    tcp_ana = true_class_prob(sonuc[ana_ad]["yt"], sonuc[ana_ad]["ypr"])
    for ad in adlar[1:]:
        d = sonuc[ad]
        dauc = paired_delta_auc(y_true, sonuc[ana_ad]["ypr"], d["ypr"],
                                B=args.bootstrap, seed=args.seed)
        tcp_oth = true_class_prob(d["yt"], d["ypr"])
        try:
            w_stat, w_p = wilcoxon(tcp_ana, tcp_oth, zero_method="wilcox",
                                   alternative="two-sided")
            w_stat, w_p = float(w_stat), float(w_p)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        karsilastirmalar[f"{ana_ad}  vs  {ad}"] = {
            "delta_auc_ana_eksi_diger": dauc,
            "wilcoxon_signed_rank_true_class_prob": {
                "statistic": w_stat, "p_value": w_p,
                "anlamli_0.05": bool(np.isfinite(w_p) and w_p < 0.05),
                "aciklama": "Örnek-başı doğru sınıf olasılığı; ANA lehine yön "
                            "medyan farkı pozitifse ANA daha iyi kalibre.",
            },
        }

    # --- Ekran özeti ---
    df = pd.DataFrame(tablo)
    print("\n" + "=" * 80)
    print("MODEL BAŞINA ROC-AUC ve DOĞRULUK (%95 Güven Aralığı — bootstrap)")
    print("=" * 80)
    for r in tablo:
        print(f"{r['Model']:<44} AUC={r['ROC-AUC']:.3f} "
              f"[{r['AUC %95 GA alt']:.3f}, {r['AUC %95 GA üst']:.3f}]  "
              f"Acc={r['Doğruluk']:.3f} "
              f"[{r['Acc %95 GA alt']:.3f}, {r['Acc %95 GA üst']:.3f}]")
    print("-" * 80)
    for isim, k in karsilastirmalar.items():
        da = k["delta_auc_ana_eksi_diger"]
        w = k["wilcoxon_signed_rank_true_class_prob"]
        print(f"{isim}")
        print(f"   ΔAUC={da['delta_auc_ortalama']:+.3f} "
              f"[{da['delta_auc_ci95'][0]:+.3f}, {da['delta_auc_ci95'][1]:+.3f}]  "
              f"bootstrap p={da['bootstrap_p']:.4f}")
        print(f"   Wilcoxon signed-rank p={w['p_value']:.4g}  "
              f"({'ANLAMLI' if w['anlamli_0.05'] else 'anlamlı değil'} @0.05)")
    print("=" * 80)

    # --- Dosyalar ---
    df.to_csv(ortak.SONUCLAR_DIR / "istatistik_analizi.csv",
              index=False, encoding="utf-8-sig", float_format="%.4f")
    with open(ortak.SONUCLAR_DIR / "istatistik_analizi.json", "w", encoding="utf-8") as f:
        json.dump({
            "aciklama": "AUC/Doğruluk %95 GA = stratified bootstrap. Karşılaştırma = "
                        "paired bootstrap ΔAUC + Wilcoxon signed-rank (true-class prob).",
            "bootstrap_tekrar": args.bootstrap,
            "seed": args.seed,
            "test_ornek_sayisi": int(len(y_true)),
            "ana_model": ana_ad,
            "model_guven_araliklari": ga,
            "eslestirilmis_karsilastirmalar": karsilastirmalar,
        }, f, indent=2, ensure_ascii=False)

    # --- Forest plot: AUC + %95 GA ---
    plt.figure(figsize=(9, 0.9 * len(adlar) + 1.5), facecolor="white")
    y = np.arange(len(adlar))[::-1]
    for yi, ad in zip(y, adlar):
        auc = ga[ad]["auc"]
        lo, hi = ga[ad]["auc_ci95"]
        renk = "#d62728" if ad == ana_ad else "#1f77b4"
        plt.errorbar(auc, yi, xerr=[[auc - lo], [hi - auc]], fmt="o",
                     color=renk, capsize=5, lw=2, markersize=8)
        plt.text(hi + 0.008, yi, f"{auc:.3f} [{lo:.3f}, {hi:.3f}]",
                 va="center", fontsize=9)
    plt.axvline(0.5, color="gray", linestyle="--", lw=1, label="Rastgele (0.50)")
    plt.yticks(y, [a.split(" (")[0] for a in adlar], fontsize=9)
    plt.xlabel("ROC-AUC (%95 güven aralığı)")
    plt.title("Modellerin ROC-AUC Değerleri ve %95 Güven Aralıkları (Test Seti)")
    plt.xlim(0.45, 1.02)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(ortak.SONUCLAR_DIR / "istatistik_auc_ci.png", dpi=300)
    plt.close()

    print(f"\n[+] Kaydedildi: istatistik_analizi.json / .csv / istatistik_auc_ci.png")


if __name__ == "__main__":
    main()
