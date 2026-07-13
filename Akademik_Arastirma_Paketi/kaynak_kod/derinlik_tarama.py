# -*- coding: utf-8 -*-
"""
derinlik_tarama.py — Kayan Pencere Konfigürasyon Taraması
=========================================================

`derinlik_analizi.py` tek konfigürasyonda (step=1, tam hacim) kayan pencere +
birleştirmenin orta-3 temelini geçemediğini gösterdi; başlıca suçlu, modelin hiç
görmediği DAĞILIM-DIŞI UÇ KESİTLERdi (`max`/`top-k` özgüllüğü çökertti). Bu script,
bunu adresleyen genişletilmiş bir tarama yapar:

  - merkez_orani : pencerelemeyi hacmin ortadaki bu oranıyla SINIRLA (uç kesitleri dışla)
                   {1.0 (tam), 0.7, 0.5, 0.3}
  - step         : {1, 2, 3}
  - method       : mean / max / top-3 / top-5

VERİMLİLİK: Her bölme (test+val) için tam step=1 pencere skorları YALNIZCA BİR KEZ
GPU'da hesaplanır; tüm konfigürasyonlar bu skorlardan alt-küme + yeniden birleştirme
ile (yeniden çıkarım YOK) türetilir. Konfig başlangıç indeksleri, `ortak`'taki tek
kaynak `window_baslangiclari` ile üretilir (select_windows ile birebir aynı).

SIZINTISIZ SEÇİM: En iyi (konfig, method) VAL ROC-AUC'ye göre seçilir; onun TEST
metrikleri raporlanır. Tüm ızgara şeffaflık için ayrıca yazdırılır/kaydedilir.

Çıktı:
  - sonuclar/derinlik_tarama.csv    (tüm konfig × method ızgarası)
  - sonuclar/derinlik_tarama.png    (ROC-AUC ısı haritası: merkez_orani × method, en iyi step)

Kullanım:
    python derinlik_tarama.py
    python derinlik_tarama.py --merkez 1.0 0.6 0.4 --step 1 2 --k 3 5
"""

import argparse
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import ortak
from degerlendirme import run_inference
from esik_analizi import metrik_hesapla
from derinlik_analizi import kayan_pencere_skorlari, youden_esigi

WINDOW = 3


def alt_kume_skorlari(full_scores, step, merkez_orani):
    """Tam step=1 pencere skorlarından bir konfigürasyonun alt-kümesini türetir.

    full_scores: step=1, tam hacim skorları (indeks = mutlak pencere başlangıcı).
    """
    n_slices = len(full_scores) + WINDOW - 1     # step=1 tam: len = N-window+1
    idx = ortak.window_baslangiclari(n_slices, window=WINDOW, step=step,
                                     merkez_orani=merkez_orani)
    return [full_scores[b] for b in idx]


def hacim_skorlari(tum_full, step, merkez_orani, method, k):
    """Tüm hacimler için: alt-küme -> birleştir -> hacim skoru dizisi."""
    return np.array([
        ortak.aggregate_scores(alt_kume_skorlari(fs, step, merkez_orani), method=method, k=k)
        for fs in tum_full
    ])


def main():
    ap = argparse.ArgumentParser(description="Kayan pencere konfigürasyon taraması")
    ap.add_argument("--merkez", type=float, nargs="+", default=[1.0, 0.7, 0.5, 0.3],
                    help="Merkez bölge oranları (1.0 = tam hacim)")
    ap.add_argument("--step", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--k", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    ortak.SONUCLAR_DIR.mkdir(parents=True, exist_ok=True)
    device = ortak.get_device()
    print(f"[*] Cihaz: {device}\n[*] Model: {ortak.MODEL_PATH.name}")
    model = ortak.load_model(device=device)

    # --- Orta-3 TEMEL (test) ---
    print("\n[1] Orta-3 temel çıkarımı (test + val)...")
    yt, _, pt_base = run_inference(model, ortak.TEST_DIR, device)
    yv, _, pv_base = run_inference(model, ortak.VAL_DIR, device)
    base_test_roc = float(roc_auc_score(yt, pt_base))
    base_test_acc = metrik_hesapla(yt, pt_base, 0.50)["accuracy"]
    base_val_roc = float(roc_auc_score(yv, pv_base))
    print(f"    Temel: test ROC-AUC {base_test_roc:.4f}, Doğruluk@0.50 {base_test_acc*100:.1f}%")

    # --- Tam step=1 pencere skorları (BİR KEZ) ---
    print("\n[2] Tam step=1 pencere skorları (test)...")
    yt2, test_full = kayan_pencere_skorlari(model, ortak.TEST_DIR, device,
                                            step=1, batch_size=args.batch_size)
    assert np.array_equal(yt, yt2)
    print("[2b] Tam step=1 pencere skorları (val)...")
    yv2, val_full = kayan_pencere_skorlari(model, ortak.VAL_DIR, device,
                                           step=1, batch_size=args.batch_size)
    assert np.array_equal(yv, yv2)

    # --- Izgara: her (merkez, step, method) için val'de seç, test'te raporla ---
    yontemler = [("mean", 0), ("max", 0)] + [("topk", k) for k in args.k]
    satirlar = []
    for merkez in args.merkez:
        for step in args.step:
            for method, k in yontemler:
                pt = hacim_skorlari(test_full, step, merkez, method, k)
                pv = hacim_skorlari(val_full, step, merkez, method, k)
                t_val = youden_esigi(yv, pv)
                m_test_050 = metrik_hesapla(yt, pt, 0.50)
                m_test_val = metrik_hesapla(yt, pt, t_val)
                # temsili pencere sayısı (ortalama)
                n_pencere = int(np.mean([len(alt_kume_skorlari(fs, step, merkez))
                                         for fs in test_full]))
                ad = ("mean" if method == "mean" else
                      "max" if method == "max" else f"top-{k}")
                satirlar.append({
                    "merkez_orani": merkez, "step": step, "method": ad,
                    "ort_pencere": n_pencere,
                    "val_roc": float(roc_auc_score(yv, pv)),
                    "test_roc": float(roc_auc_score(yt, pt)),
                    "test_pr": float(average_precision_score(yt, pt)),
                    "test_acc_050": m_test_050["accuracy"],
                    "test_sens_050": m_test_050["duyarlilik"],
                    "test_spec_050": m_test_050["ozgulluk"],
                    "test_acc_valesik": m_test_val["accuracy"],
                })

    df = pd.DataFrame(satirlar)

    # --- Sızıntısız seçim: val_roc en yükseği ---
    en_iyi = df.loc[df["val_roc"].idxmax()]

    # --- Rapor ---
    pd.set_option("display.width", 220)
    print("\n" + "=" * 110)
    print(f"KAYAN PENCERE TARAMASI — tüm ızgara (test n={len(yt)}). "
          f"Temel: test ROC-AUC {base_test_roc:.4f}, Doğruluk {base_test_acc*100:.1f}%")
    print("=" * 110)
    goster = df.copy()
    goster["Δroc"] = goster["test_roc"] - base_test_roc
    goster["Δacc"] = goster["test_acc_050"] - base_test_acc
    print(goster[["merkez_orani", "step", "method", "ort_pencere", "val_roc",
                  "test_roc", "Δroc", "test_acc_050", "test_spec_050",
                  "test_acc_valesik", "Δacc"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))
    print("=" * 110)

    print(f"\n[VAL'DE SEÇİLEN EN İYİ] merkez_orani={en_iyi['merkez_orani']}, "
          f"step={int(en_iyi['step'])}, method={en_iyi['method']} "
          f"(ort. {int(en_iyi['ort_pencere'])} pencere)")
    print(f"  val ROC-AUC : {en_iyi['val_roc']:.4f}  (temel val {base_val_roc:.4f})")
    print(f"  TEST ROC-AUC: {en_iyi['test_roc']:.4f}  (temel {base_test_roc:.4f}, "
          f"Δ{en_iyi['test_roc']-base_test_roc:+.4f})")
    print(f"  TEST Doğruluk@0.50   : {en_iyi['test_acc_050']*100:.1f}%  "
          f"(temel {base_test_acc*100:.1f}%, Δ{(en_iyi['test_acc_050']-base_test_acc)*100:+.1f} puan)")
    print(f"  TEST Doğruluk@val-eşik: {en_iyi['test_acc_valesik']*100:.1f}%")

    # En yüksek TEST ROC-AUC (üst sınır referansı; seçim değil)
    en_test = df.loc[df["test_roc"].idxmax()]
    yener = en_iyi["test_roc"] > base_test_roc + 0.01 or \
        en_iyi["test_acc_050"] > base_test_acc + 0.01
    print(f"\n[KARAR] Val-seçili konfig temeli "
          f"{'ANLAMLI biçimde GEÇİYOR' if yener else 'anlamlı biçimde GEÇEMİYOR'} "
          f"(eşik: ΔROC>+0.01 veya ΔAcc>+1 puan).")
    print(f"        (Referans — ızgaradaki en yüksek TEST ROC-AUC: "
          f"{en_test['test_roc']:.4f} @ merkez={en_test['merkez_orani']}, "
          f"step={int(en_test['step'])}, {en_test['method']})")

    # --- ÖNERİLEN ayarı UI/CLI için kaydet (esik_ayari.json kalıbı) ---
    ei_pt = hacim_skorlari(test_full, int(en_iyi["step"]), en_iyi["merkez_orani"],
                           "mean" if en_iyi["method"] == "mean" else
                           ("max" if en_iyi["method"] == "max" else "topk"),
                           int(en_iyi["method"].split("-")[1]) if en_iyi["method"].startswith("top") else 0)
    with open(ortak.KAYAN_AYAR_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "method": "mean" if en_iyi["method"] == "mean" else
                      ("max" if en_iyi["method"] == "max" else "topk"),
            "k": int(en_iyi["method"].split("-")[1]) if en_iyi["method"].startswith("top") else 0,
            "merkez_orani": float(en_iyi["merkez_orani"]),
            "step": int(en_iyi["step"]),
            "kriter": "en yüksek val ROC-AUC (sızıntısız); merkez bölge = uç kesitleri dışlar",
            "test_roc": float(en_iyi["test_roc"]),
            "val_roc": float(en_iyi["val_roc"]),
            "test_metrikleri": metrik_hesapla(yt, ei_pt, 0.50),
            "temel_test_roc": base_test_roc,
            "temel_test_acc": base_test_acc,
        }, f, indent=2, ensure_ascii=False)

    # --- Dosyalar ---
    df.to_csv(ortak.SONUCLAR_DIR / "derinlik_tarama.csv", index=False,
              encoding="utf-8-sig", float_format="%.4f")
    with open(ortak.SONUCLAR_DIR / "derinlik_tarama.json", "w", encoding="utf-8") as f:
        json.dump({
            "temel": {"test_roc": base_test_roc, "test_acc_050": base_test_acc,
                      "val_roc": base_val_roc},
            "secim_kriteri": "en yüksek val_roc (sızıntısız); test'te raporlandı",
            "en_iyi_val_secili": en_iyi.to_dict(),
            "en_yuksek_test_roc": en_test.to_dict(),
            "izgara": satirlar,
        }, f, indent=2, ensure_ascii=False)

    # --- Isı haritası: merkez_orani × method (her hücrede en iyi step'in test ROC'u) ---
    methods = ["mean", "max"] + [f"top-{k}" for k in args.k]
    merkezler = sorted(set(df["merkez_orani"]), reverse=True)
    M = np.full((len(merkezler), len(methods)), np.nan)
    for i, mo in enumerate(merkezler):
        for j, me in enumerate(methods):
            alt = df[(df["merkez_orani"] == mo) & (df["method"] == me)]
            if len(alt):
                M[i, j] = alt["test_roc"].max()
    fig, ax = plt.subplots(figsize=(1.6 * len(methods) + 2, 1.1 * len(merkezler) + 2),
                           facecolor="white")
    im = ax.imshow(M, cmap="RdYlGn", vmin=base_test_roc - 0.06, vmax=base_test_roc + 0.03,
                   aspect="auto")
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods)
    ax.set_yticks(range(len(merkezler)))
    ax.set_yticklabels([f"merkez {int(m*100)}%" for m in merkezler])
    for i in range(len(merkezler)):
        for j in range(len(methods)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_title(f"Kayan Pencere TEST ROC-AUC (hücre = en iyi step)\n"
                 f"Temel orta-3 = {base_test_roc:.3f} (yeşil=temelin üstü)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="TEST ROC-AUC")
    plt.tight_layout()
    plt.savefig(ortak.SONUCLAR_DIR / "derinlik_tarama.png", dpi=200)
    plt.close()

    print(f"\n[+] Kaydedildi: derinlik_tarama.csv / .json / .png  ->  {ortak.SONUCLAR_DIR}")


if __name__ == "__main__":
    main()
