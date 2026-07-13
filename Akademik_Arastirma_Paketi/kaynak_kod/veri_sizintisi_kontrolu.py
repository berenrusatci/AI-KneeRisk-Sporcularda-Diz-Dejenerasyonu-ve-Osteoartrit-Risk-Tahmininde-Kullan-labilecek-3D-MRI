# -*- coding: utf-8 -*-
"""
veri_sizintisi_kontrolu.py — Veri Sızıntısı (Data Leakage) Kontrolü
===================================================================

Eğitim (train), doğrulama (val) ve test setleri arasında herhangi bir örtüşme
olup olmadığını İKİ bağımsız yöntemle kesin olarak doğrular:

  1) KİMLİK BAZLI: Her hacim klasörünün adı (örn. 'normal-3DESS-128-64_sample_0004')
     benzersiz bir örnek kimliğidir. Split'ler arası isim kesişimi hesaplanır.
     (İndeksler sınıf içinde global ve ayrık olduğundan bu geçerli bir testtir.)

  2) İÇERİK BAZLI (piksel): Her hacimden temsili kesitler (0, 16, 32, 48, 63)
     okunur ve birleşik MD5 özeti çıkarılır. Aynı özet iki farklı split'te
     görülürse — klasör adları farklı olsa bile — bu gerçek bir piksel-düzeyi
     sızıntısıdır. İçerik bazlı yöntem, yeniden adlandırılmış kopyaları da yakalar.

Çıktı: sonuclar/veri_sizintisi_raporu.txt

Kullanım:
    python veri_sizintisi_kontrolu.py
"""

import os
import glob
import hashlib
from itertools import combinations

import ortak

# İçerik özeti için okunacak temsili kesit indeksleri (64 kesitlik hacimlerde)
TEMSILI_INDEKSLER = [0, 16, 32, 48, 63]


def hacimleri_topla(split_dir):
    """split -> {tam_klasor_adi: klasor_yolu} (tüm sınıflar birleşik)."""
    hacimler = {}
    for cls in ortak.CLASSES:
        cdir = os.path.join(str(split_dir), cls)
        if not os.path.isdir(cdir):
            continue
        for ad in os.listdir(cdir):
            yol = os.path.join(cdir, ad)
            if os.path.isdir(yol):
                # Sınıf ön ekiyle benzersiz kimlik (normal/abnormal ayrı isim uzayı)
                hacimler[f"{cls}/{ad}"] = yol
    return hacimler


def hacim_icerik_ozeti(hacim_yolu):
    """Temsili kesitlerin baytlarından birleşik MD5 üretir."""
    pngler = sorted(glob.glob(os.path.join(hacim_yolu, "*.png")),
                    key=lambda p: ortak.natural_sort_key(os.path.basename(p)))
    if not pngler:
        return None
    md5 = hashlib.md5()
    n = len(pngler)
    secilen = [pngler[i] for i in TEMSILI_INDEKSLER if i < n] or pngler
    for p in secilen:
        with open(p, "rb") as f:
            md5.update(f.read())
    return md5.hexdigest()


def main():
    ortak.SONUCLAR_DIR.mkdir(parents=True, exist_ok=True)
    splitler = {"train": ortak.TRAIN_DIR, "val": ortak.VAL_DIR, "test": ortak.TEST_DIR}

    satirlar = []
    def yaz(s=""):
        print(s)
        satirlar.append(s)

    yaz("=" * 64)
    yaz("VERİ SIZINTISI (DATA LEAKAGE) KONTROL RAPORU")
    yaz("=" * 64)

    # --- Hacimleri topla ---
    hacim_map = {k: hacimleri_topla(v) for k, v in splitler.items()}
    for ad, m in hacim_map.items():
        yaz(f"[*] {ad:5s}: {len(m)} hacim")
    yaz("")

    # --- 1) KİMLİK BAZLI KESİŞİM ---
    yaz("-" * 64)
    yaz("1) KİMLİK (KLASÖR ADI) BAZLI KESİŞİM")
    yaz("-" * 64)
    kimlik_temiz = True
    for a, b in combinations(splitler.keys(), 2):
        ortak_kimlik = set(hacim_map[a]) & set(hacim_map[b])
        durum = "TEMİZ ✓" if not ortak_kimlik else f"SIZINTI ✗ ({len(ortak_kimlik)} örtüşme)"
        yaz(f"    {a:5s} ∩ {b:5s} : {durum}")
        if ortak_kimlik:
            kimlik_temiz = False
            for x in list(ortak_kimlik)[:10]:
                yaz(f"        - {x}")
    yaz("")

    # --- 2) İÇERİK BAZLI (PİKSEL HASH) ---
    yaz("-" * 64)
    yaz("2) İÇERİK (PİKSEL HASH) BAZLI KESİŞİM")
    yaz(f"    (temsili kesitler: {TEMSILI_INDEKSLER})")
    yaz("-" * 64)
    yaz("    [*] Hacim içerik özetleri hesaplanıyor...")

    # split -> {hash: [kimlik,...]}
    hash_map = {}
    for split, m in hacim_map.items():
        h2ids = {}
        for kimlik, yol in m.items():
            h = hacim_icerik_ozeti(yol)
            if h is None:
                continue
            h2ids.setdefault(h, []).append(kimlik)
        hash_map[split] = h2ids

    icerik_temiz = True
    for a, b in combinations(splitler.keys(), 2):
        ortak_hash = set(hash_map[a]) & set(hash_map[b])
        durum = "TEMİZ ✓" if not ortak_hash else f"SIZINTI ✗ ({len(ortak_hash)} özdeş içerik)"
        yaz(f"    {a:5s} ∩ {b:5s} : {durum}")
        if ortak_hash:
            icerik_temiz = False
            for h in list(ortak_hash)[:10]:
                yaz(f"        - {hash_map[a][h]}  <=>  {hash_map[b][h]}")

    # Split içi tekrar (bilgi amaçlı)
    yaz("")
    yaz("    Split-içi özdeş içerik (bilgi amaçlı):")
    for split, h2ids in hash_map.items():
        tekrar = {h: ids for h, ids in h2ids.items() if len(ids) > 1}
        yaz(f"        {split:5s}: {len(tekrar)} tekrarlanan içerik grubu")
    yaz("")

    # --- SONUÇ ---
    yaz("=" * 64)
    yaz("SONUÇ")
    yaz("=" * 64)
    if kimlik_temiz and icerik_temiz:
        yaz("✓ HİÇBİR VERİ SIZINTISI TESPİT EDİLMEDİ.")
        yaz("  Train / val / test setleri hem kimlik hem piksel düzeyinde")
        yaz("  tamamen ayrıktır. Test sonuçları güvenilirdir.")
    else:
        yaz("✗ DİKKAT: Olası veri sızıntısı tespit edildi (yukarıya bakınız).")
    yaz("=" * 64)

    rapor_yolu = ortak.SONUCLAR_DIR / "veri_sizintisi_raporu.txt"
    with open(rapor_yolu, "w", encoding="utf-8") as f:
        f.write("\n".join(satirlar) + "\n")
    print(f"\n[+] Rapor kaydedildi: {rapor_yolu}")


if __name__ == "__main__":
    main()
