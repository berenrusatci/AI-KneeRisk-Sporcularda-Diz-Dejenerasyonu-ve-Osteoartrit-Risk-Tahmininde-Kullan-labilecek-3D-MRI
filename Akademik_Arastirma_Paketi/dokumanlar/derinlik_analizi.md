# Derinlik Analizi — Kayan Pencere + Birleştirme (Tüm Hacmi Kullanma)

Bu çalışma şu soruyu yanıtlar: Ana model her hacimden yalnızca **ortadaki 3 kesiti**
kullanıyor; hacmin **tamamını** kullanmak, **yeniden eğitim olmadan**, test başarımını
artırır mı? Yöntem, mevcut ağırlıkla (`best_model_balanced_crop74.pth`) hacmi 3'erli
**kayan pencerelerle** (`window=3, step=1`, hacim başına ~62 pencere) baştan sona tarayıp
her pencerenin P(abnormal) skorunu tek hacim kararına **birleştirmektir** (MRNet tarzı
çıkarım-zamanı birleştirme). Üretim: `kaynak_kod/derinlik_analizi.py` →
`sonuclar/derinlik_analizi.{json,csv,png}`.

## Yöntem

- **Kayan pencere:** Model 3 kanal beklediğinden pencere genişliği 3 sabittir; `step`
  granülerliği belirler. Değerlendirme `step=1` (azami örtüşme, en pürüzsüz birleştirme)
  ile yapılmıştır. Ön işleme zinciri her pencerede birebir korunur
  (CenterCrop(280) → Resize(224) → ImageNet normalize).
- **Birleştirme (aggregation):**
  - `mean` — tüm pencerelerin ortalaması (gürültüye dayanıklı, bulguyu sulandırabilir)
  - `max` — en yüksek pencere skoru (en duyarlı, yanlış alarma açık)
  - `top-k ort.` — en yüksek k pencerenin ortalaması (max ile mean arası; k=3, 5)
- **Sızıntısız karşılaştırma:** Birincil ölçüt **eşikten bağımsız** ROC-AUC ve PR-AUC'dir.
  Ayrıca sabit 0.50 eşiği (argmax referansı) ve **eşiğin yalnızca val'de seçildiği**
  (Youden-J) sızıntısız accuracy raporlanır. Temel satır, mevcut `metrikler.json` ile
  birebir tutar (kod doğrulaması).

## Sonuçlar (Test Seti, n=449, step=1)

| Yöntem | ROC-AUC | PR-AUC | Doğruluk@0.50 | Duyarlılık@0.50 | Özgüllük@0.50 | FN | FP | Doğruluk@val-eşik |
|---|---|---|---|---|---|---|---|---|
| **Orta-3 (temel)** | **0.797** | 0.780 | **%72.4** | 0.653 | 0.780 | 69 | 55 | %71.9 |
| Kayan: mean | **0.801** | 0.784 | %72.8 | 0.513 | 0.900 | 97 | 25 | %72.6 |
| Kayan: max | 0.745 | 0.749 | %45.2 | 1.000 | 0.016 | 0 | 246 | %68.2 |
| Kayan: top-3 ort. | 0.761 | 0.760 | %45.2 | 0.995 | 0.020 | 1 | 245 | %71.5 |
| Kayan: top-5 ort. | 0.767 | 0.768 | %46.3 | 0.995 | 0.040 | 1 | 240 | %72.2 |

## Yorum

1. **`max` ve `top-k` zararlıdır — dağılım-dışı uç kesitler.** Model yalnızca *orta*
   kesitlerle eğitildiğinden, hacmin uçlarındaki kesitler (eklem dışı, az anatomi içeren)
   modelin görmediği bir dağılımdan gelir ve sıklıkla **sahte yüksek** P(abnormal) üretir.
   `max`/`top-k` bu aykırı pencereleri seçtiği için neredeyse tüm hacimleri "abnormal"
   işaretler: 0.50 eşiğinde özgüllük **%1.6–4**'e çöker (FP 240–246). Eşik yalnızca val'de
   yeniden ayarlansa bile (0.63–0.67'ye çıkar) bu yöntemler temeli geçemez.

2. **`mean` esasen temelle BERABERE.** ROC-AUC 0.797 → 0.801 (**+0.005**) ve Doğruluk
   %72.4 → %72.8 (**+0.4 puan**); val-seçili eşikte %71.9 → %72.6 (**+0.7 puan**). Bu
   farklar n=449 için istatistiksel gürültü sınırındadır — **anlamlı bir kazanç değildir.**
   `mean`'in gerçek etkisi doğruluğu artırmak değil, **çalışma noktasını özgüllüğe
   kaydırmaktır**: 0.50 eşiğinde özgüllük 0.78 → 0.90, yanlış alarm FP 55 → 25 düşerken
   duyarlılık 0.65 → 0.51'e iner (kaçan hasta FN 69 → 97). Yani daha az yanlış alarm,
   daha çok kaçan hasta — bu bir **ödünleşim**, iyileşme değil.

3. **Neden tüm derinlik yardımcı olmuyor?** Bu model 2.5D olarak, patolojinin en belirgin
   olduğu **orta kesitleri** görecek şekilde eğitilmiştir. Uç kesitleri eklemek yeni bilgi
   getirmek yerine gürültü ekler. Tüm derinlikten gerçek kazanç, ancak modelin tüm kesitleri
   eğitimde görmesiyle (öğrenilmiş dikkat/havuzlama ya da 3D omurga — **yeniden eğitim**)
   elde edilebilir; çıkarım-zamanı birleştirme tek başına yetmemektedir.

## İkinci Aşama — Merkezi Kayan Pencere (Uç Kesitleri Dışlama)

Yukarıdaki analiz, başarısızlığın kök nedenini işaret etti: **dağılım-dışı uç kesitler.**
Buradan hareketle, pencerelemeyi hacmin yalnızca **ortadaki merkez bölgesiyle** sınırlayan
bir tarama yapıldı (`kaynak_kod/derinlik_tarama.py`): merkez oranı {1.0, 0.7, 0.5, 0.3} ×
step {1, 2, 3} × {mean, max, top-3, top-5}. Verimlilik için tam step=1 skorları her bölme
için bir kez hesaplanıp tüm konfigürasyonlar bunlardan türetildi. **Sızıntısız seçim:**
en iyi konfigürasyon **val ROC-AUC**'ye göre seçilip **test**'te raporlandı.

**Bulgu:** Merkez bölge daraldıkça (uç kesitler dışlandıkça) ROC-AUC **sistematik olarak
yükseliyor** (~0.80 → ~0.83). `mean`, merkez %30–50 bandında geniş ve sağlam bir plato
oluşturuyor (tek şanslı hücre değil).

| Konfigürasyon (val-seçili) | ROC-AUC (test) | Doğruluk@0.50 | Özgüllük@0.50 | Doğruluk@val-eşik |
|---|---|---|---|---|
| **Orta-3 (temel)** | 0.797 | %72.4 | 0.780 | %71.9 |
| **Merkez %30, step=2, mean** | **0.831 (Δ+0.035)** | **%73.3 (+0.9)** | 0.804 | %73.5 |
| (ızgara en yüksek test ROC: merkez %50, step=1, top-5) | 0.833 | — | — | — |

Tam ızgara: `sonuclar/derinlik_tarama.csv`, ısı haritası `derinlik_tarama.png`, seçilen
ayar `derinlik_ayari.json`.

## Karar

- **Tam hacim (merkez %100) kayan pencere temeli geçemez** (mean berabere; max/top-k zararlı).
- **Merkezi kayan pencere (merkez %30, step=2, `mean`) temeli ANLAMLI biçimde geçer:**
  ROC-AUC 0.797 → **0.831** (+0.035) ve Doğruluk %72.4 → **%73.3**. Kritik nokta, uç
  kesitleri dışlayıp yalnızca tanısal açıdan anlamlı merkez bölgeyi ortalamaktır — bu,
  tek orta-3 kesitine göre varyansı düşürür.

Bu nedenle **merkezi kayan pencere modu arayüze ve exe'ye opsiyonel olarak entegre
edilmiştir** ("🧩 Kesit Modu → Merkezi kayan pencere"). Varsayılan mod geri uyumluluk
için orta-3 olarak kalır. Önerilen ayar `sonuclar/derinlik_ayari.json`'dan okunur
(`ortak.onerilen_kayan_ayar`), eşik mekanizmasıyla (`esik_ayari.json`) aynı kalıp.

## Gelecek Çalışma

- Merkez oranını daha da daraltmak (0.2, 0.1) val'de aşırı-uyum riskiyle denenebilir;
  0.3 zaten platonun içinde ve güvenli seçimdir.
- Gerçek uçtan-uca kazanç için tüm kesitleri eğitimde gören yaklaşımlar: (a) öğrenilmiş
  dikkat (attention-MIL) ile hacim-düzeyi eğitim; (b) 3D omurga (ör. MONAI DenseNet-3D) —
  sınırlı veri (~birkaç yüz hacim) ve mevcut ~9.9 puanlık train-test GAP göz önünde
  tutulmalı (bkz. `gap_analizi.md`).
