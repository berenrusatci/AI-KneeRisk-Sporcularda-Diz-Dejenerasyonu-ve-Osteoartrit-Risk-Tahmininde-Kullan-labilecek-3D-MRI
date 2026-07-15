# -*- coding: utf-8 -*-
"""
risk_skorlama.py — Risk Skorlama Sistemi (0-100) ve Risk Grupları
==================================================================

Başvuru Hedef 5'in uygulaması: modelin P(abnormal) çıktısını 0-100 arası bir
"diz dejenerasyonu / osteoartrit risk skoruna" dönüştürür ve her hacmi
**Düşük (0-33) / Orta (34-66) / Yüksek (67-100)** risk grubuna atar
(bkz. ortak.risk_skoru / ortak.risk_grubu).

Skorlama sistemi bağımsız TEST setinde doğrulanır: her risk grubundaki
GERÇEK hasta (abnormal) oranı hesaplanır. Geçerli bir risk skorunda bu oran
Düşük -> Orta -> Yüksek doğrultusunda **artmalıdır** (monotonluk = ayrım gücü).

Çıktı:
  - sonuclar/risk_skorlama.json     (grup özetleri + monotonluk + skor kalibrasyonu)
  - sonuclar/risk_skorlama.csv      (her hacim için skor, grup, gerçek etiket)
  - sonuclar/risk_skorlama.png      (skor histogramı + grup başına gerçek hasta oranı)

Kullanım:
    python risk_skorlama.py
"""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import ortak
from degerlendirme import run_inference


def grup_ozetleri(y_true, skorlar, gruplar):
    """Her risk grubu için: adet, gerçek hasta oranı (prevalans), ortalama skor."""
    ozet = []
    for ad in ortak.RISK_GRUP_ADLARI:
        maske = np.array([g == ad for g in gruplar])
        n = int(maske.sum())
        if n == 0:
            ozet.append({"grup": ad, "adet": 0, "gercek_hasta_orani": None,
                         "ortalama_skor": None, "hasta_sayisi": 0})
            continue
        hasta = int(y_true[maske].sum())              # abnormal = 1
        ozet.append({
            "grup": ad,
            "adet": n,
            "hasta_sayisi": hasta,
            "gercek_hasta_orani": float(hasta / n),   # bu gruptaki gerçek abnormal oranı
            "ortalama_skor": float(np.mean(skorlar[maske])),
        })
    return ozet


def main():
    ortak.SONUCLAR_DIR.mkdir(parents=True, exist_ok=True)
    device = ortak.get_device()
    model = ortak.load_model(device=device)
    print(f"[*] Cihaz: {device}\n[*] Model: {ortak.MODEL_PATH.name}")

    print("[*] Test çıkarımı...")
    y_true, _, p_abnormal = run_inference(model, ortak.TEST_DIR, device)
    p_abnormal = np.asarray(p_abnormal)

    skorlar = np.array([ortak.risk_skoru(p) for p in p_abnormal])
    gruplar = [ortak.risk_grubu(s) for s in skorlar]

    ozet = grup_ozetleri(y_true, skorlar, gruplar)

    # Monotonluk kontrolü: Düşük < Orta < Yüksek gerçek hasta oranı?
    oranlar = [o["gercek_hasta_orani"] for o in ozet if o["gercek_hasta_orani"] is not None]
    monoton = all(oranlar[i] <= oranlar[i + 1] for i in range(len(oranlar) - 1))

    # Skor-olasılık ilişkisi (skor = round(p*100) olduğundan birebir; yine de raporlanır)
    korelasyon = float(np.corrcoef(skorlar, y_true)[0, 1])

    # --- Ekran raporu ---
    print("\n" + "=" * 68)
    print("RİSK SKORLAMA SİSTEMİ — TEST SETİ DOĞRULAMASI")
    print("=" * 68)
    print(f"{'Risk Grubu':<12}{'Adet':>7}{'Gerçek Hasta':>15}{'Hasta Oranı':>14}{'Ort. Skor':>12}")
    for o in ozet:
        if o["adet"] == 0:
            print(f"{o['grup']:<12}{o['adet']:>7}{'-':>15}{'-':>14}{'-':>12}")
        else:
            print(f"{o['grup']:<12}{o['adet']:>7}{o['hasta_sayisi']:>15}"
                  f"{o['gercek_hasta_orani']*100:>13.1f}%{o['ortalama_skor']:>12.1f}")
    print("-" * 68)
    print(f"Monotonluk (Düşük<=Orta<=Yüksek hasta oranı): "
          f"{'EVET (geçerli)' if monoton else 'HAYIR'}")
    print(f"Skor-etiket korelasyonu (Pearson r): {korelasyon:.3f}")
    print("=" * 68)

    # --- Dosyalar ---
    per_sample = pd.DataFrame({
        "risk_skoru": skorlar,
        "risk_grubu": gruplar,
        "p_abnormal": p_abnormal,
        "gercek_etiket": [ortak.CLASSES[int(y)] for y in y_true],
    })
    per_sample.to_csv(ortak.SONUCLAR_DIR / "risk_skorlama.csv",
                      index=False, encoding="utf-8-sig", float_format="%.4f")

    with open(ortak.SONUCLAR_DIR / "risk_skorlama.json", "w", encoding="utf-8") as f:
        json.dump({
            "aciklama": "Risk skoru = round(P(abnormal)*100). Gruplar: Düşük 0-33, "
                        "Orta 34-66, Yüksek 67-100. Metrikler bağımsız test setinde.",
            "grup_sinirlari": {"dusuk_ust": ortak.RISK_GRUP_SINIRLARI[0],
                               "orta_ust": ortak.RISK_GRUP_SINIRLARI[1]},
            "test_ornek_sayisi": int(len(y_true)),
            "gruplar": ozet,
            "monotonluk_gecerli": bool(monoton),
            "skor_etiket_korelasyonu_pearson": korelasyon,
        }, f, indent=2, ensure_ascii=False)

    # --- Grafik: (sol) skor histogramı gerçek etikete göre, (sağ) grup hasta oranı ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor="white")

    binler = np.arange(0, 101, 10)
    ax1.hist(skorlar[y_true == 0], bins=binler, alpha=0.6, color="#1f77b4",
             label="Gerçek: Normal", edgecolor="white")
    ax1.hist(skorlar[y_true == 1], bins=binler, alpha=0.6, color="#d62728",
             label="Gerçek: Abnormal", edgecolor="white")
    for x in ortak.RISK_GRUP_SINIRLARI:
        ax1.axvline(x + 0.5, color="gray", linestyle="--", lw=1)
    ax1.set_xlabel("Risk Skoru (0-100)")
    ax1.set_ylabel("Hacim sayısı")
    ax1.set_title("Risk Skoru Dağılımı (gerçek etikete göre)")
    ax1.legend()
    ax1.text(16, ax1.get_ylim()[1] * 0.92, "Düşük", ha="center", color="gray")
    ax1.text(50, ax1.get_ylim()[1] * 0.92, "Orta", ha="center", color="gray")
    ax1.text(83, ax1.get_ylim()[1] * 0.92, "Yüksek", ha="center", color="gray")

    adlar = [o["grup"] for o in ozet]
    oran_pct = [(o["gercek_hasta_orani"] or 0) * 100 for o in ozet]
    renkler = ["#2ca02c", "#ff7f0e", "#d62728"]
    barlar = ax2.bar(adlar, oran_pct, color=renkler, edgecolor="black", alpha=0.85)
    for b, o in zip(barlar, ozet):
        if o["adet"]:
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                     f"{o['hasta_sayisi']}/{o['adet']}", ha="center", fontsize=10)
    ax2.set_ylabel("Gruptaki gerçek hasta (abnormal) oranı (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("Risk Grubu ile Gerçek Hasta Oranı\n(monotonluk = ayrım gücü)")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(ortak.SONUCLAR_DIR / "risk_skorlama.png", dpi=300)
    plt.close()

    print(f"\n[+] Kaydedildi: risk_skorlama.json / .csv / .png")


if __name__ == "__main__":
    main()
