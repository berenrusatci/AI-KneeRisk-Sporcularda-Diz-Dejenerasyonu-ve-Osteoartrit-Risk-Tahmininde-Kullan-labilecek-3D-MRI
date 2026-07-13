# Ablasyon Analizi (Ablation Study)

Bu çalışma, "hangi mimari omurga ve eğitim stratejisi diz MR verisinde daha iyi
genelliyor?" sorusunu yanıtlar. Kayıtlı checkpoint'ler **aynı bağımsız test seti**
(n=449) üzerinde, **her biri kendi mimarisi ve kendi eğitim ön işlemesiyle**
değerlendirilmiştir (adil karşılaştırma). Üretim: `kaynak_kod/model_karsilastirma.py`
→ `sonuclar/model_karsilastirma.csv`, `model_karsilastirma.png`.

## Karşılaştırılan Konfigürasyonlar

| Model | Omurga | Kayıp / Strateji | Ön işleme |
|---|---|---|---|
| **ANA** | ConvNeXt-Tiny | Focal + sınıf dengesi + 2 fazlı | CenterCrop(280)+Resize |
| Önceki ConvNeXt | ConvNeXt-Tiny | Focal (önceki sürüm) | CenterCrop(280)+Resize |
| EfficientNet-B0 | EfficientNet-B0 | CrossEntropy | Resize(224) |
| ResNet50 | ResNet50 | CrossEntropy (transfer) | Resize(224) |

## Sonuçlar (Test Seti)

| Model | Doğruluk | Makro-F1 | ROC-AUC | Duyarlılık | Özgüllük |
|---|---|---|---|---|---|
| **ConvNeXt-Tiny (Focal+Dengeli+Crop) [ANA]** | **%72.4** | **0.718** | **0.797** | 0.653 | 0.780 |
| ConvNeXt-Tiny (Focal, önceki) | %67.5 | 0.675 | 0.761 | 0.779 | 0.592 |
| EfficientNet-B0 (CrossEntropy) | %65.9 | 0.629 | 0.700 | 0.422 | 0.848 |
| ResNet50 (Transfer, CrossEntropy) | %59.7 | 0.594 | 0.658 | 0.774 | 0.456 |

## Yorum — Bileşen Katkıları

1. **Omurga seçimi (ConvNeXt vs EfficientNet-B0 vs ResNet50):**
   ConvNeXt tabanlı modeller, aynı 2.5D girdide EfficientNet-B0 ve ResNet50'yi
   Doğruluk, F1 ve AUC'de belirgin biçimde geçmiştir. ResNet50 (%59.7) neredeyse
   çoğunluk-tahmini seviyesinde kalmış; ConvNeXt'in modern tasarımı bu domainde
   daha güçlü özellikler çıkarmıştır. **En büyük tekil katkı omurga seçimidir.**

2. **Focal Loss + sınıf dengesi + CenterCrop (ANA vs önceki ConvNeXt):**
   Aynı omurgada, dengeleme + kırpma iyileştirmeleri Doğruluğu %67.5 → %72.4,
   AUC'yi 0.761 → 0.797'ye çıkarmıştır. Ayrıca önceki sürüm yüksek duyarlılık /
   düşük özgüllük (0.779 / 0.592) ile "aşırı temkinli" (çok yanlış-pozitif) iken,
   ANA model **duyarlılık-özgüllük dengesini** iyileştirmiştir (0.653 / 0.780).

3. **Duyarlılık-özgüllük dengesi:** EfficientNet-B0 yüksek özgüllük ama çok düşük
   duyarlılık (0.422) sergileyerek hasta hacimlerini kaçırmaktadır; ResNet50 ise
   tersine düşük özgüllük göstermektedir. ANA model her iki metrikte de dengeli ve
   en yüksek AUC'ye sahip olarak en tutarlı sınıflandırıcıdır.

## Eğitim vs Test Karşılaştırması (ANA model)

Ablasyona ek olarak, ANA modelin train/val/test üçlüsündeki başarımı
(`gap_overfitting_analizi.py`):

| Split | Doğruluk | Makro-F1 | ROC-AUC |
|---|---|---|---|
| Eğitim (train) | %82.3 | 0.821 | 0.914 |
| Doğrulama (val) | %74.8 | 0.746 | 0.826 |
| Test | %72.4 | 0.718 | 0.797 |

Detaylı overfitting/GAP yorumu için `gap_analizi.md`.

## Kapsam Notu

MONAI **DenseNet-3D** (tam hacim, tek kanal) ve **R(2+1)D** (video) modelleri farklı
giriş biçimleri kullandığından bu birleşik 2.5D karşılaştırmasına dahil edilmemiştir;
adil karşılaştırma için ayrı, kendi ön işleme boru hatlarıyla değerlendirilmeleri
**gelecek çalışma** olarak önerilir.
