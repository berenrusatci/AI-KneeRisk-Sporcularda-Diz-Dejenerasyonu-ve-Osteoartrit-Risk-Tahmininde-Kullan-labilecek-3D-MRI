# -*- coding: utf-8 -*-
"""
tahmin.py — Komut Satırı Tek Hacim Çıkarımı
===========================================

Bir hacim klasörü (içinde en az 3 PNG kesit) verilerek modelin tahminini
ve sınıf olasılıklarını yazdırır. Çekirdek mantık ortak.predict_volume'dedir;
bu dosya yalnızca kullanışlı bir komut satırı sarmalayıcısıdır (UI ile aynı API).

Kullanım:
    python tahmin.py "C:\\...\\data\\test\\abnormal\\abnormal-3DESS-128-64_sample_0001"
    python tahmin.py "<hacim>" --yuksek-duyarlilik      # önerilen düşük eşik
    python tahmin.py "<hacim>" --esik 0.40              # özel eşik
"""

import os
import sys
import argparse

import ortak


def main():
    parser = argparse.ArgumentParser(description="Diz MR tek hacim tahmini")
    parser.add_argument("hacim_yolu", help="Kesit PNG'lerini içeren hacim klasörü")
    parser.add_argument("--model", default=str(ortak.MODEL_PATH), help="Model .pth yolu")
    parser.add_argument("--esik", type=float, default=None,
                        help="Karar eşiği P(abnormal). Düşük eşik = yüksek duyarlılık.")
    parser.add_argument("--yuksek-duyarlilik", action="store_true",
                        help="esik_analizi.py'nin önerdiği düşük eşiği kullan")
    args = parser.parse_args()

    # Eşik önceliği: --esik > --yuksek-duyarlilik > standart (0.50)
    if args.esik is not None:
        esik, mod = args.esik, "özel"
    elif args.yuksek_duyarlilik:
        esik, bilgi = ortak.onerilen_esik()
        mod = "yüksek duyarlılık (önerilen)" if bilgi else "yüksek duyarlılık (esik_ayari.json yok, 0.50)"
    else:
        esik, mod = ortak.DEFAULT_THRESHOLD, "standart"

    device = ortak.get_device()
    model = ortak.load_model(model_path=args.model, device=device)
    r = ortak.predict_volume(args.hacim_yolu, model=model, device=device, threshold=esik)

    print("=" * 50)
    print(f"Hacim   : {args.hacim_yolu}")
    print(f"Cihaz   : {device}")
    print(f"Eşik    : {esik:.3f} ({mod})")
    print("-" * 50)
    print(f"TAHMİN  : {r['pred_name']}")
    print(f"Güven   : %{r['confidence']*100:.2f}")
    print(f"Olasılıklar -> normal: %{r['probs']['normal']*100:.2f} | "
          f"abnormal: %{r['probs']['abnormal']*100:.2f}")
    print(f"Risk Skoru : {r['risk_skoru']}/100  ->  {r['risk_grubu']} risk grubu")
    print(f"Kullanılan kesitler: {[os.path.basename(p) for p in r['slice_files']]}")
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[HATA] {e}")
        sys.exit(1)
