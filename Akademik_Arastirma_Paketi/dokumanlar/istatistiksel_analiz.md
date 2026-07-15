# İstatistiksel Anlamlılık ve Güven Aralıkları

Bu doküman, başvuru Yöntem bölümünde belirtilen istatistiksel gereklerin
uygulamasını raporlar: (1) ROC-AUC ve Doğruluk için **%95 güven aralıkları**,
(2) ANA model ile diğer mimariler arasında **eşleştirilmiş (paired) anlamlılık
testleri** (Wilcoxon signed-rank + paired bootstrap ΔAUC). Üretim:
`kaynak_kod/istatistik_analizi.py` → `sonuclar/istatistik_analizi.{json,csv}`,
`istatistik_auc_ci.png`.

## Yöntem

- **%95 güven aralığı (GA):** Stratified bootstrap (sınıf oranını koruyan yeniden
  örnekleme), B=2000 tekrar. AUC ve Doğruluk için 2.5–97.5 persentil aralığı.
- **ΔAUC (paired bootstrap):** Aynı bootstrap örneklemleri iki modele de
  uygulanır (eşleştirilmiş); ANA − diğer AUC farkının %95 GA'sı ve iki-yönlü
  bootstrap p-değeri hesaplanır.
- **Wilcoxon signed-rank testi** (Wilcoxon, 1945): Örnek-başı "doğru sınıfa atanan
  olasılık" (`y=1 → p`, `y=0 → 1−p`; yüksek = iyi) üzerinde ANA vs diğer eşleştirilmiş
  karşılaştırma. Tüm modeller **aynı** test setinde, aynı sırayla değerlendirildiği
  için örnekler eşleştirilmiştir.

## Sonuçlar — Model Başına %95 GA (Test Seti, n=449)

| Model | ROC-AUC [%95 GA] | Doğruluk [%95 GA] |
|---|---|---|
| **ConvNeXt-Tiny (Focal+Dengeli+Crop) — ANA** | **0.797 [0.751, 0.837]** | **0.724 [0.682, 0.768]** |
| ConvNeXt-Tiny (Focal, önceki) | 0.761 [0.712, 0.803] | 0.675 [0.633, 0.715] |
| EfficientNet-B0 (CrossEntropy) | 0.700 [0.649, 0.751] | 0.659 [0.621, 0.699] |
| ResNet50 (Transfer, CrossEntropy) | 0.658 [0.610, 0.707] | 0.597 [0.555, 0.641] |

Forest plot: `sonuclar/istatistik_auc_ci.png`.

## Sonuçlar — ANA Model vs Diğerleri (Eşleştirilmiş)

| Karşılaştırma | ΔAUC (ANA−diğer) [%95 GA] | Bootstrap p | Wilcoxon p | Anlamlı (@0.05) |
|---|---|---|---|---|
| ANA vs önceki ConvNeXt | +0.036 [+0.005, +0.068] | 0.021 | 3.9×10⁻⁷ | ✅ |
| ANA vs EfficientNet-B0 | +0.096 [+0.046, +0.147] | <0.001 | 2.9×10⁻⁴ | ✅ |
| ANA vs ResNet50 | +0.138 [+0.084, +0.191] | <0.001 | 0.040 | ✅ |

## Yorum

- **ANA modelin üstünlüğü istatistiksel olarak anlamlıdır:** Üç karşılaştırmada da
  ΔAUC'nin %95 güven aralığı **sıfırı içermez** (yani AUC farkı 0'dan anlamlı
  biçimde büyüktür) ve Wilcoxon signed-rank testi p < 0.05 verir.
- **En güçlü fark ResNet50'ye karşıdır** (ΔAUC +0.138), en dar fark ise aynı
  omurgadaki önceki ConvNeXt sürümünedir (ΔAUC +0.036) — yine de anlamlıdır; bu,
  Focal + sınıf dengesi + CenterCrop iyileştirmelerinin ölçülebilir katkısını
  doğrular (bkz. `ablasyon_analizi.md`).
- **Güven aralıkları**, n=449'luk test setinde nokta tahminlerinin belirsizliğini
  şeffaf biçimde ortaya koyar: ANA modelin AUC'si %95 olasılıkla ~0.75–0.84
  bandındadır.

## Yeniden Üretim

```
python istatistik_analizi.py --bootstrap 2000 --seed 42
```
