# TÜBİTAK 2209-A — Sonuç Raporu

**Proje:** AI-KneeRisk — Sporcularda Diz Dejenerasyonu ve Osteoartrit Risk Tahmininde Kullanılabilecek 3D/2.5D MRI Tabanlı Derin Öğrenme Modeli
**Program:** TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projeleri Destekleme Programı
**Alan:** Spor ve Sağlık · Diz MR Görüntülerinde Normal / Abnormal Ayrımı

> ⚠️ **Sorumluluk Reddi:** Bu çalışma akademik/araştırma amaçlıdır. Geliştirilen model bir **klinik tanı aracı değildir** ve tıbbi karar için tek başına kullanılamaz. Model, sertifikalı bir tıbbi cihaz değil, bir karar-destek prototipidir.

> **Not:** Bu belge, proje kapsamında yazılmış tüm teknik dokümanların (`README`, `metodoloji`, `model_karti`, `ablasyon_analizi`, `hiperparametre_analizi`, `esik_analizi`, `gap_analizi`, `derinlik_analizi`, `OverFitting_DataLeakage`, `KULLANIM`) birleştirilmiş sonuç raporudur. Tüm sayısal sonuçlar bağımsız test seti (n=449) üzerinde ilgili scriptler çalıştırılarak üretilmiştir; uydurma değer yoktur.

---

## 1. Yönetici Özeti

Bu araştırmada, diz MR hacimlerinin **normal (sağlıklı)** mı yoksa **abnormal (patolojik)** mi olduğunu otomatik sınıflandıran, hesaplama açısından verimli **2.5D ConvNeXt-Tiny** tabanlı bir derin öğrenme modeli geliştirilmiştir. Model, ön-tarama (triyaj) amaçlı ikili bir görüntü sınıflandırıcısıdır.

**Bağımsız test setinde (n=449) öne çıkan sonuçlar:**

| Metrik | Değer |
|---|---|
| Doğruluk (Accuracy) | **%72.4** |
| Duyarlılık (Sensitivity) | %65.3 |
| Özgüllük (Specificity) | %78.0 |
| ROC-AUC | **0.797** |
| PR-AUC (Average Precision) | 0.780 |
| Makro-F1 | 0.718 |
| Genelleme Boşluğu (GAP, train−test) | 9.9 puan (hafif overfitting) |

Ek olarak **Yüksek Duyarlılık Modu** (karar eşiği 0.50 → 0.43) ile duyarlılık **%81.9**'a çıkarılabilmekte, tarama bağlamında kaçan hasta sayısı 69 → 36'ya düşmektedir. Model; veri sızıntısı denetimi, overfitting/GAP analizi, mimari ablasyonu, hiperparametre taraması, karar eşiği optimizasyonu, kesit-derinliği analizi ve Grad-CAM açıklanabilirliği ile bütünsel olarak doğrulanmış; ayrıca kurulum gerektirmeyen bir Windows demo uygulamasına (**DizMR.exe**) dönüştürülmüştür.

---

## 2. Veri Seti ve Ön İşleme

### 2.1 Veri Seti

- **Modalite:** 3D-DESS (Double Echo Steady State) diz MR sekansı.
- **Yapı:** Her hasta/örnek bir **hacim**tir; 64 adet ardışık eksenel kesitten oluşur (`slice_00.png … slice_63.png`), gri tonlama.
- **Bölünme (hacim/hasta düzeyinde, sızıntısız):**

  | Split | Normal | Abnormal | Toplam hacim |
  |---|---|---|---|
  | train | 1161 | 921 | 2082 |
  | val | 248 | 197 | 445 |
  | test | 250 | 199 | 449 |

- **Sınıf dengesi:** ≈ %56 normal / %44 abnormal (ılımlı dengesizlik); Focal Loss ve sınıf ağırlıklarıyla ele alınmıştır.

### 2.2 Ön İşleme Hattı

- **Boyutlandırma/Formatlama:** Görüntüler PNG'ye dönüştürülmüş, bozuk dosyalar temizlenmiştir.
- **Min-Max Normalizasyonu:** Cihaz/hasta kaynaklı parlaklık tutarsızlıklarını gidermek için yoğunluk değerleri ölçeklenmiştir.
- **Kontrast İyileştirme (CLAHE):** Doku sınırlarının (kıkırdak, lezyon, kemik) daha iyi öğrenilebilmesi için Contrast Limited Adaptive Histogram Equalization uygulanmıştır.
- **Değerlendirme dönüşüm zinciri (eğitim ve testte birebir aynı):** `CenterCrop(280) → Resize(224,224) → Normalize(ImageNet mean/std)`.
- **Yalnızca eğitimde augmentasyon:** RandomHorizontalFlip, RandomAffine(±10°, öteleme/ölçek), ColorJitter, RandomErasing.

---

## 3. Metodoloji

### 3.1 2.5D Yaklaşımı

Tam 3D konvolüsyonun yüksek donanım maliyeti ve 2D modellerin derinlik bilgisinden yoksunluğu göz önüne alınarak **2.5D** temsili benimsenmiştir:

1. Hacimden ortadaki 3 kesit seçilir (`[mid-1, mid, mid+1]`) — patolojinin en görünür olduğu merkezî bölge.
2. Kesitler gri tonlamaya çevrilip 3 kanal olarak yığılır → `[3, 224, 224]`. (Kanallar RGB değil, komşu kesitlerdir; ince yönde bağlam kazandırır.)

### 3.2 Model Mimarisi

- **Omurga:** `torchvision.models.convnext_tiny`, ImageNet ön-eğitimli (~28M parametre).
- **Sınıflandırma başı:** `classifier[2]` → `Sequential(Dropout(0.5), Linear(768, 2))`.
- **Sınıflar:** `0 = normal`, `1 = abnormal`.

### 3.3 Eğitim Stratejisi — İki Fazlı Transfer Öğrenme

| Faz | Epoch | Eğitilen | Optimizer | LR |
|---|---|---|---|---|
| 1 — Isınma | 10 | Yalnızca baş (omurga donuk) | AdamW (wd=1e-2) | 1e-3 |
| 2 — İnce ayar | ≤40 | Tüm ağ | AdamW (wd=1e-2) | 5e-5 + CosineAnnealing |

- **Kayıp:** Focal Loss (γ=2.0), sınıf ağırlığı `[1.0, 1.25]`.
- **Erken durdurma:** val loss 8 tur iyileşmezse durur; en iyi checkpoint korunur.
- **AMP:** float16 karışık hassasiyet (hız + VRAM).

Faz 1'in tek başına yetersiz (underfitting), Faz 2 tam ince ayarın başarımı açtığı hem hiperparametre taramasında hem model karşılaştırmasında ampirik olarak gözlenmiştir.

### 3.4 Değerlendirme Protokolü

Test seti (n=449) üzerinde argmax tahmini ve softmax olasılıkları toplanır. Raporlanan metrikler: Accuracy, sınıf bazlı Precision/Recall/F1, Duyarlılık, Özgüllük, Makro/Ağırlıklı F1 ve **eşikten bağımsız** ROC-AUC + PR-AUC (pozitif sınıf = abnormal).

---

## 4. Ana Sonuçlar (Model Kartı)

**Ana model:** `best_model_balanced_crop74.pth` — ConvNeXt-Tiny 2.5D, Focal Loss + sınıf dengesi + merkez kırpma.

### 4.1 Test Metrikleri (n=449)

| Metrik | Değer |
|---|---|
| Doğruluk | %72.4 |
| Duyarlılık | %65.3 |
| Özgüllük | %78.0 |
| ROC-AUC | 0.797 |
| PR-AUC | 0.780 |
| Makro-F1 | 0.718 |

**Sınıf bazlı:**

| Sınıf | Precision | Recall | F1 | Destek |
|---|---|---|---|---|
| Normal (Sağlıklı) | 0.739 | 0.780 | 0.759 | 250 |
| Abnormal (Hasta) | 0.703 | 0.653 | 0.677 | 199 |

**Karışıklık matrisi:** TN=195, FP=55, FN=69, TP=130.

---

## 5. Doğrulama Analizleri

### 5.1 Veri Sızıntısı (Data Leakage) Kontrolü

İki bağımsız yöntemle denetlenmiştir:
1. **Kimlik bazlı:** train/val/test hacim kimlikleri (klasör adları) arasında kesişim **YOK**.
2. **İçerik bazlı:** Temsili kesitlerin (0,16,32,48,63) MD5 özetleri üzerinden setler arası özdeş içerik ve split-içi tekrar **YOK**.

**Sonuç:** `train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅`. Test başarımı gerçek, sızıntısız genellemeyi yansıtır.

> **Erken not (öğrenme eğrileri):** Val başarısının Train'i geçmesi anomalisi sızıntıdan değil, (a) eğitimde aktif olan **Dropout(0.5)**'in val'de kapanmasından ve (b) yalnızca eğitim setine uygulanan **augmentasyonun** eğitim işini zorlaştırmasından kaynaklanır.

### 5.2 GAP (Genelleme Boşluğu) ve Overfitting

Ana model, augmentasyonsuz deterministik ön işlemeyle üç split'te değerlendirilmiştir:

| Split | n | Doğruluk | Makro-F1 | ROC-AUC |
|---|---|---|---|---|
| Eğitim (train) | 2082 | %82.3 | 0.821 | 0.914 |
| Doğrulama (val) | 445 | %74.8 | 0.746 | 0.826 |
| Test | 449 | %72.4 | 0.718 | 0.797 |

| Boşluk | Değer |
|---|---|
| GAP Doğruluk (train − test) | **9.9 puan** |
| GAP Doğruluk (train − val) | 7.5 puan |
| GAP ROC-AUC (train − test) | 0.117 |

**Yorum:** GAP ≈ %9.9 "küçük–orta" aralıkta olup **hafif overfitting**e işaret eder; tıbbi görüntüleme literatüründe kabul edilebilir düzeydedir. Val (%74.8) ve test (%72.4) yakınlığı, val setinin güvenilir bir vekil olduğunu ve model seçiminin sağlıklı yapıldığını gösterir. Overfitting'i sınırlayan tasarım kararları: Dropout 0.5, augmentasyon, AdamW ağırlık çürümesi (wd=1e-2), erken durdurma (patience=8), transfer öğrenme.

### 5.3 Mimari Ablasyonu

Aynı bağımsız test setinde, her model kendi mimarisi ve ön işlemesiyle değerlendirilmiştir:

| Model | Doğruluk | Makro-F1 | ROC-AUC | Duyarlılık | Özgüllük |
|---|---|---|---|---|---|
| **ConvNeXt-Tiny (Focal+Dengeli+Crop) — ANA** | **%72.4** | **0.718** | **0.797** | 0.653 | 0.780 |
| ConvNeXt-Tiny (Focal, önceki) | %67.5 | 0.675 | 0.761 | 0.779 | 0.592 |
| EfficientNet-B0 (CrossEntropy) | %65.9 | 0.629 | 0.700 | 0.422 | 0.848 |
| ResNet50 (Transfer, CrossEntropy) | %59.7 | 0.594 | 0.658 | 0.774 | 0.456 |

**Bileşen katkıları:**
1. **En büyük tekil katkı omurga seçimidir:** ConvNeXt, aynı 2.5D girdide EfficientNet-B0 ve ResNet50'yi belirgin biçimde geçmiştir (ResNet50 neredeyse çoğunluk-tahmini seviyesinde kalmıştır).
2. **Focal + sınıf dengesi + CenterCrop:** Aynı omurgada Doğruluğu %67.5 → %72.4, AUC'yi 0.761 → 0.797'ye çıkarmış ve duyarlılık-özgüllük dengesini iyileştirmiştir.
3. ANA model, her iki metrikte dengeli ve en yüksek AUC'ye sahip en tutarlı sınıflandırıcıdır.

> **Kapsam notu:** MONAI DenseNet-3D (tam hacim) ve R(2+1)D (video) farklı giriş biçimleri kullandığından bu birleşik 2.5D karşılaştırmasına dâhil edilmemiş, gelecek çalışma olarak önerilmiştir.

### 5.4 Hiperparametre Analizi

**Dokümante seçimler (nihai model):** ConvNeXt-Tiny, 2.5D orta-3 kesit, batch=16, Focal Loss (γ=2.0), sınıf ağırlığı [1.0, 1.25], 2 fazlı transfer (Faz1 lr=1e-3 → Faz2 lr=5e-5 + Cosine), AdamW (wd=1e-2), erken durdurma (patience=8), Dropout 0.5, AMP.

**Ampirik mini-tarama** (özellik-önbellekli linear-probe; amaç göreli etki):

- **Öğrenme oranı:** 1e-2 (%69.0) ≈ 1e-3 (%68.3) > 1e-4 (%65.6 — underfitting). Seçilen 1e-3, tam ince ayar için muhafazakâr/güvenli.
- **Kayıp:** CrossEntropy (%69.0) ≈ Focal (%68.3) linear-probe'da eşit; Focal'in faydası **tam ince ayarlı + dengesiz** rejimde ortaya çıkar.
- **Sınıf ağırlığı:** [1,1] (%69.2) > [1,1.25] (%68.3) > [1,1.6] (%67.9); ağırlık 1.6 → 1.25'e düşürülmesi, saf doğruluk yerine **dengeli hata profili** için bilinçli bir tercihtir.

En büyük başarım sıçraması hiperparametre ince ayarından çok **omurga seçimi ve tam ince ayardan** gelmektedir.

### 5.5 Karar Eşiği (Threshold) Analizi — Duyarlılık Artırma

Eşikler **yalnızca val setinde** seçilip **test setinde** raporlanmıştır (test sızıntısı önlenmiştir):

| Senaryo | Eşik | Doğruluk | Duyarlılık | Özgüllük | Kaçan hasta (FN) | Yanlış alarm (FP) |
|---|---|---|---|---|---|---|
| Standart (0.50) | 0.500 | %72.4 | %65.3 | %78.0 | 69 | 55 |
| Youden-J (val) | 0.517 | %71.9 | %59.3 | %82.0 | 81 | 45 |
| **Duyarlılık ≥ 0.85 — ÖNERİLEN** | **0.426** | %68.8 | **%81.9** | %58.4 | **36** | 104 |
| Duyarlılık ≥ 0.90 | 0.398 | %64.4 | %86.9 | %46.4 | 26 | 134 |

**Yorum:** Önerilen eşik (0.43) duyarlılığı %65.3 → %81.9'a çıkarır, kaçan hasta 69 → 36'ya düşer; bedeli özgüllüğün %78 → %58'e inmesidir. Tarama bağlamında savunulabilir bir ödünleşimdir (yanlış negatif, yanlış pozitiften klinik olarak daha maliyetlidir). Eşik modelin ağırlıklarını değiştirmez; yalnızca operasyon noktasını kaydırır ve dağıtım anında serbestçe ayarlanabilir. Arayüz ve CLI'da "Yüksek Duyarlılık" modu olarak sunulur.

### 5.6 Kesit-Derinliği Analizi — Kayan Pencere

**Soru:** Yeniden eğitim olmadan hacmin tamamını kullanmak başarımı artırır mı?

**Tam hacim (merkez %100) kayan pencere:** `max`/`top-k` birleştirme **zararlıdır** — model yalnızca orta kesitlerle eğitildiğinden uç kesitler dağılım-dışıdır ve sahte yüksek P(abnormal) üretir (özgüllük %1.6–4'e çöker). `mean` esasen temelle **beraberedir** (ROC-AUC 0.797 → 0.801; anlamlı değil), yalnızca çalışma noktasını özgüllüğe kaydırır.

**Merkezi kayan pencere (uç kesitleri dışlama):** Merkez bölge daraldıkça ROC-AUC sistematik yükselir:

| Konfigürasyon (val-seçili) | ROC-AUC (test) | Doğruluk@0.50 |
|---|---|---|
| Orta-3 (temel) | 0.797 | %72.4 |
| **Merkez %30, step=2, mean** | **0.831 (Δ+0.035)** | **%73.3 (+0.9)** |

**Karar:** Merkezi kayan pencere (merkez %30, step=2, `mean`) temeli **anlamlı biçimde geçer** ve arayüze/exe'ye opsiyonel mod olarak entegre edilmiştir. Varsayılan mod geri uyumluluk için orta-3 kalır.

### 5.7 Açıklanabilirlik (Grad-CAM + SHAP)

İki tamamlayıcı XAI yöntemi uygulanmıştır (bkz. `dokumanlar/aciklanabilirlik_xai.md`):
- **Grad-CAM:** Modelin son özellik bloğu (`features[-1]`) üzerinden üretilen ısı haritaları, kararların ağırlıklı olarak **diz ekleminin merkezî yapılarına** dayandığını gösterir.
- **SHAP** (`shap.GradientExplainer`): Piksel düzeyinde imzalı katkı haritaları (kırmızı: abnormal olasılığını artıran, mavi: azaltan bölgeler). SHAP, Grad-CAM ile **aynı anatomik bölgeleri** işaret ederek bulguyu bağımsız bir teorik temelde (Shapley değerleri) doğrular.

İki bağımsız yöntemin aynı bölgeleri işaret etmesi, modelin klinik güvenilirliğini artırır. Çıktılar: `sonuclar/gradcam_ornekleri/`, `sonuclar/shap_ornekleri/`.

### 5.8 İstatistiksel Anlamlılık ve Güven Aralıkları

ROC-AUC ve Doğruluk için **%95 güven aralıkları** (stratified bootstrap, B=2000) ve ANA model ile diğer mimariler arasında **eşleştirilmiş anlamlılık testleri** (Wilcoxon signed-rank + paired bootstrap ΔAUC) hesaplanmıştır (bkz. `dokumanlar/istatistiksel_analiz.md`):

| Model | ROC-AUC [%95 GA] | Doğruluk [%95 GA] |
|---|---|---|
| **ANA (ConvNeXt-Tiny)** | **0.797 [0.751, 0.837]** | **0.724 [0.682, 0.768]** |
| önceki ConvNeXt | 0.761 [0.712, 0.803] | 0.675 [0.633, 0.715] |
| EfficientNet-B0 | 0.700 [0.649, 0.751] | 0.659 [0.621, 0.699] |
| ResNet50 | 0.658 [0.610, 0.707] | 0.597 [0.555, 0.641] |

**ANA modelin üstünlüğü istatistiksel olarak anlamlıdır:** üç karşılaştırmada da ΔAUC'nin %95 GA'sı sıfırı içermez ve Wilcoxon p < 0.05'tir (ANA vs önceki: ΔAUC +0.036, p=3.9×10⁻⁷; vs EfficientNet: +0.096, p=2.9×10⁻⁴; vs ResNet50: +0.138, p=0.040).

### 5.9 Risk Skorlama Sistemi (0-100)

Model çıktısı `P(abnormal)`, 0-100 arası bir **diz dejenerasyonu / OA risk skoruna** dönüştürülür ve hacim **Düşük (0-33) / Orta (34-66) / Yüksek (67-100)** grubuna atanır (bkz. `dokumanlar/risk_skorlama.md`). Test setinde grupların gerçek hasta oranı monoton artar; skorlamanın ayrım gücünü doğrular:

| Risk Grubu | Adet | Gerçek Hasta Oranı |
|---|---|---|
| 🟢 Düşük | 66 | %13.6 |
| 🟡 Orta | 329 | %43.2 |
| 🔴 Yüksek | 54 | %88.9 |

Risk skoru CLI (`tahmin.py`) ve arayüzde gösterilir; monotonluk geçerli, skor–etiket Pearson r = 0.519.

---

## 6. Erken Aşama Denemeleri

Nihai modele ulaşmadan önce çok sayıda mimari denenmiştir: MONAI DenseNet, ConvNeXt-Tiny 2.5D, EfficientNet-B0/B3, DenseNet121 2.5D, R(2+1)D, ResNet50 3D/2.5D, VGG16 2.5D, ViT. Ön analizlerde **%69 val doğruluğu** ile öne çıkan iki model (ConvNeXt-Tiny 2.5D ve MONAI DenseNet) üzerinde optimizasyon yürütülmüş; nihai ana model **ConvNeXt-Tiny 2.5D** olmuştur.

**DenseNet3D erken denemesi:** 20 epoch eğitimde 6. epoch civarı optimum, 13. epoch'tan itibaren overfitting gözlenmiş; checkpoint sistemiyle en iyi ağırlıklar korunmuştur. Focal Loss olmadan val doğruluğu %52.6 → %43.2'ye düşmüş (sınıf dengesizliği etkisi), 3D veri yükü batch=8'de M4 Max GPU'da OOM vermiştir.

---

## 7. Ürün / Dağıtım — DizMR

Model, kurulum gerektirmeyen bir Windows uygulamasına (**DizMR.exe**, PyInstaller) ve bir **Streamlit demo arayüzüne** dönüştürülmüştür:

- Test setinden hacim seçme veya kendi PNG kesitlerini yükleme.
- Sonuçta sınıf olasılıkları, modele giren 3 kesit ve Grad-CAM ısı haritası gösterilir.
- Yan panelden **karar eşiği** (standart 0.50 / yüksek duyarlılık 0.43 / özel) ve **kesit modu** (orta-3 / merkezi kayan pencere) seçilebilir.
- CLI: `python tahmin.py "<hacim>" --yuksek-duyarlilik` veya `--esik 0.40`.

Dağıtım paketi (`DizMR_Windows_x64.zip`, ~330 MB) Releases sayfasında yayımlanır; öz-kontrol için `DizMR.exe --selftest` sağlanmıştır.

---

## 8. Sınırlamalar ve Etik

- **Duyarlılık standart eşikte görece düşük (%65.3):** 69 yanlış negatif; eşik ayarıyla %81.9'a çıkarılabilir (bkz. §5.5).
- **Tek merkez / tek sekans (3D-DESS):** Farklı cihaz, protokol veya popülasyonda başarım düşebilir (dağılım kayması). **Dış geçerleme yapılmamıştır.**
- **2.5D yaklaşımı** yalnızca merkez kesitleri kullanır; hacmin tamamındaki bilgi kısmen göz ardı edilir.
- **Hafif overfitting** (GAP ≈ 9.9 puan) mevcuttur.
- Model çıktıları **klinik doğrulamadan geçmemiştir**; klinik tanı aracı değildir.

---

## 9. Sonuç ve Gelecek Çalışma

Geliştirilen ConvNeXt-Tiny 2.5D modeli, sızıntısız bağımsız test setinde **%72.4 doğruluk ve 0.797 ROC-AUC** ile diz MR hacimlerinde normal/abnormal ayrımını başarıyla gerçekleştirmiş; karar eşiği ayarıyla tarama senaryolarına uyarlanabilir bir duyarlılık profili sunmuştur. Tüm sonuçlar veri sızıntısı, GAP, ablasyon, hiperparametre, eşik ve derinlik analizleriyle doğrulanmış, Grad-CAM ile açıklanabilir kılınmış ve çalıştırılabilir bir demoya dönüştürülmüştür.

**Gelecek çalışma yönleri:**
- Daha fazla/çeşitli veri ve daha güçlü düzenlileştirme ile GAP'in azaltılması.
- Tüm kesitleri eğitimde gören yaklaşımlar: öğrenilmiş dikkat (attention-MIL) ile hacim-düzeyi eğitim veya 3D omurga (ör. MONAI DenseNet-3D).
- Farklı merkez/cihaz verisiyle **dış geçerleme**.
- Sporcu popülasyonuna transfer öğrenme ile uyarlanma (2209-A ötesi genişletme).

---

## Ek — Kaynak Doküman Dizini

Bu rapor aşağıdaki proje dosyalarının birleşiminden oluşturulmuştur:

| Kaynak | İçerik |
|---|---|
| `README.md` (kök) | Proje genel tanıtımı, pipeline, ana sonuçlar, depo yapısı |
| `Akademik_Arastirma_Paketi/README.md` | Paket kılavuzu, kurulum, çalıştırma |
| `dokumanlar/metodoloji.md` | Veri, ön işleme, mimari, eğitim, değerlendirme |
| `dokumanlar/model_karti.md` | Model kartı, test sonuçları, sınırlamalar |
| `dokumanlar/ablasyon_analizi.md` | Mimari karşılaştırması |
| `dokumanlar/hiperparametre_analizi.md` | Hiperparametre seçimleri + ampirik tarama |
| `dokumanlar/esik_analizi.md` | Karar eşiği / duyarlılık analizi |
| `dokumanlar/gap_analizi.md` | GAP, overfitting, veri sızıntısı |
| `dokumanlar/derinlik_analizi.md` | Kayan pencere / kesit-derinliği |
| `dokumanlar/aciklanabilirlik_xai.md` | Grad-CAM + SHAP açıklanabilirlik |
| `dokumanlar/risk_skorlama.md` | Risk skoru (0-100) + Düşük/Orta/Yüksek gruplar |
| `dokumanlar/istatistiksel_analiz.md` | %95 güven aralıkları + Wilcoxon signed-rank |
| `OverFitting_DataLeakage.md` | Erken aşama overfitting/sızıntı notları |
| `dagitim_exe/KULLANIM.md` | DizMR.exe kullanım kılavuzu |
