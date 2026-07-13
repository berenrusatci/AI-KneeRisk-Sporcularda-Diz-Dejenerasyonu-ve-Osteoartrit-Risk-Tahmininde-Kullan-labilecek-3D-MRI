# Metodoloji

Bu doküman, diz MR normal/abnormal sınıflandırma modelinin veri, ön işleme,
mimari, eğitim ve değerlendirme metodolojisini akademik bir çerçevede açıklar.

## 1. Problem Tanımı

Amaç, bir diz MR hacminin **normal (sağlıklı)** mı yoksa **abnormal (patolojik)**
mi olduğunu otomatik olarak sınıflandırmaktır. Bu, ikili (binary) bir görüntü
sınıflandırma problemidir ve spor/sağlık bağlamında ön-tarama (triyaj) amaçlıdır.

## 2. Veri Seti

- **Modalite:** 3D-DESS (Double Echo Steady State) diz MR sekansı.
- **Yapı:** Her hasta/örnek bir **hacim**tir ve 64 adet ardışık eksenel kesitten
  oluşur (`slice_00.png … slice_63.png`), her kesit 256×256, gri tonlama.
- **Bölünme:** Hacim düzeyinde önceden ayrılmış train/val/test (≈%70/15/15):

  | Split | Normal | Abnormal | Toplam hacim |
  |---|---|---|---|
  | train | 1161 | 921 | 2082 |
  | val | 248 | 197 | 445 |
  | test | 250 | 199 | 449 |

- **Sınıf dengesizliği:** ≈ %56/44; Focal Loss ve sınıf ağırlıklarıyla ele alınmıştır.

### 2.1 Veri Sızıntısı Kontrolü

Train/val/test setleri iki bağımsız yöntemle denetlenmiştir
(`veri_sizintisi_kontrolu.py`):
1. **Kimlik bazlı:** Hacim klasör adları benzersiz kimliktir; setler arası kesişim boştur.
2. **İçerik bazlı:** Her hacmin temsili kesitlerinden (0,16,32,48,63) MD5 özeti
   çıkarılır; setler arası özdeş içerik yoktur, split-içi tekrar da yoktur.

**Sonuç:** Hiçbir sızıntı yoktur; test başarımı gerçek genellemeyi yansıtır.

## 3. Ön İşleme — "2.5D" Yaklaşımı

Tam 3D konvolüsyon yerine hesaplama açısından verimli **2.5D** temsili kullanılır:

1. Hacimden ortadaki 3 kesit seçilir: `[mid-1, mid, mid+1]` (patolojinin en görünür
   olduğu merkezî bölge).
2. Her kesit gri tonlamaya çevrilir ve 3 kanal olarak yığılır → `[3, H, W]`.
   (Kanallar RGB değil, komşu kesitlerdir; böylece ince yönde bağlam kazanılır.)
3. Dönüşüm zinciri (eğitim ve testte **birebir aynı**, deterministik kısımda):
   - `CenterCrop(280)` — kenar boşluklarını/cihaz yazılarını kırpar. (Kaynak kesitler
     256×256 olduğundan bu adım pratikte simetrik sıfır-dolgu uygular; model bu
     girdiyi gördüğünden değerlendirmede korunması zorunludur.)
   - `Resize(224, 224)` — omurga giriş boyutu.
   - `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])` — ImageNet istatistikleri.
4. Yalnızca **eğitimde** ek augmentasyon: RandomHorizontalFlip, RandomAffine(±10°,
   öteleme/ölçek), ColorJitter, RandomErasing.

## 4. Model Mimarisi

- **Omurga:** `torchvision.models.convnext_tiny`, ImageNet ön-eğitimli. ConvNeXt,
  modern evrişimsel bir mimaridir; medikal 2.5D girdilerde iyi transfer özelliği
  sunar.
- **Sınıflandırma başı:** Orijinal `classifier[2]` şu şekilde değiştirilir:
  `Sequential(Dropout(0.5), Linear(768, 2))`.

## 5. Eğitim Stratejisi

**İki fazlı transfer öğrenme:**

| Faz | Epoch | Eğitilen | Optimizer | LR |
|---|---|---|---|---|
| 1 — Isınma | 10 | Yalnızca baş (omurga donuk) | AdamW (wd=1e-2) | 1e-3 |
| 2 — İnce ayar | ≤40 | Tüm ağ | AdamW (wd=1e-2) | 5e-5 + CosineAnnealing |

- **Kayıp:** Focal Loss (γ=2.0), sınıf ağırlığı `[1.0, 1.25]`. Focal Loss, kolay
  örneklerin katkısını `(1−p_t)^γ` ile azaltarak zor/az örneklenen sınıfa odaklanır.
- **Erken durdurma:** val loss 8 tur iyileşmezse durur; en iyi checkpoint korunur.
- **AMP:** float16 karışık hassasiyet (hız + VRAM).

Faz 1'in yalnızca-baş eğitiminin tek başına yetersiz kaldığı (underfitting),
Faz 2 tam ince ayarının başarımı açtığı hem hiperparametre taramasında hem de
model karşılaştırmasında ampirik olarak gözlenmiştir (bkz. ilgili dokümanlar).

## 6. Değerlendirme Protokolü

- Test seti (n=449) üzerinde **argmax** tahmini ve **softmax olasılıkları** toplanır
  (`degerlendirme.py`).
- Raporlanan metrikler: Accuracy, sınıf bazlı Precision/Recall/F1, Duyarlılık
  (abnormal recall), Özgüllük (normal recall), Makro/Ağırlıklı F1.
- **Eşikten bağımsız** metrikler: ROC eğrisi + AUC ve Precision-Recall eğrisi + AP,
  pozitif sınıf = abnormal olacak şekilde hesaplanır.
- Tüm metrikler `sonuclar/metrikler.json`'a, grafikler PNG olarak kaydedilir.

## 7. Açıklanabilirlik

Modelin son özellik bloğu (`features[-1]`) üzerinden **Grad-CAM** ısı haritaları
üretilir (`gradcam.py`). Bu, kararın hangi anatomik bölgeye dayandığını görselleştirir
ve klinik güven açısından önemlidir.

## 8. Yeniden Üretilebilirlik

Tüm ön işleme, mimari ve yollar tek bir merkezi modülde (`kaynak_kod/ortak.py`)
tanımlıdır; değerlendirme ve analiz scriptleri bunu içe aktarır. Böylece eğitimde
kullanılan ön işleme ile değerlendirmedeki ön işleme arasında tutarlılık garanti
edilir.
