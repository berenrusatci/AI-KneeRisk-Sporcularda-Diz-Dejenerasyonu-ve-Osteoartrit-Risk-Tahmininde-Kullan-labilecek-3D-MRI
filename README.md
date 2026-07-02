# TÜBİTAK 2209-A: Sporcularda Diz Dejenerasyonu ve Osteoartrit Risk Tahmini

Bu depo, sporcularda diz dejenerasyonu ve osteoartrit riskini 3D ve 2.5D Manyetik Rezonans (MR) görüntüleri üzerinden derin öğrenme teknikleriyle tahmin etmeyi amaçlayan projenin model eğitim, veri ön işleme ve performans analizi süreçlerini içermektedir.

## 🛠️ Veri Ön İşleme Hattı (Pipeline)

Modelin öznitelik çıkarımını maksimize etmek ve veri tutarlılığını sağlamak için aşağıdaki adımlar uygulanmıştır:

*   **Boyutlandırma ve Formatlama:** Tüm görüntüler **256x256 piksel** boyutuna eşitlenmiş ve kayıpsız PNG formatına dönüştürülmüştür. Bozuk veya açılamayan dosyalar veri setinden temizlenmiştir.
*   **Veri Bölümlemesi:** Veri seti; *Train (Eğitim)*, *Test* ve *Real-world (Gerçek Dünya)* olmak üzere üç bağımsız parçaya ayrılmıştır.
*   **Min-Max Normalizasyonu:** Farklı cihazlar ve hastalardan kaynaklanan parlaklık tutarsızlıklarını gidermek için görüntü yoğunluk değerleri (intensity values) 0-255 aralığına ölçeklenmiştir.
*   **Kontrast İyileştirme (CLAHE):** Doku sınırlarının (kıkırdak, lezyon, kemik vb.) model tarafından daha iyi öğrenilebilmesi için *Contrast Limited Adaptive Histogram Equalization* algoritması uygulanmıştır.

## 🧠 Model Araştırması ve 2.5D Yaklaşımı

Doğrudan 3D modellerin yüksek donanım gereksinimleri ve 2D modellerin derinlik bilgisinden yoksun olması göz önüne alınarak **2.5D yaklaşımı** benimsenmiştir. Derinlik bilgisini modele katabilmek için 3 ardışık kesit (frame), 2D modellere RGB matrisi (3 kanal) olarak beslenmiştir.

**Denenen Mimariler:**
Aşağıdaki modeller üzerinde Train/Test/Val için Loss ve Accuracy (Doğruluk) değerleri izlenmiş, nihai karar **Validation Loss ve Validation Accuracy** metriklerine göre verilmiştir:
*   MONAI DenseNet
*   ConvNeXt Tiny 2.5D
*   EfficientNet-B0 & EfficientNet-B3
*   DenseNet121 2.5D
*   R(2+1)D
*   ResNet50 3D & ResNet50 2.5D
*   VGG16 2.5D
*   ViT (Vision Transformer)

## 🚀 Seçilen ve Geliştirilen Modeller

Ön analizler sonucunda **%69 Validation Accuracy** değerine ulaşan iki model üzerinde hiperparametre optimizasyonları ve derin ince ayarlar (deep fine-tuning) devam etmektedir:

### 1. ConvNeXt Tiny 2.5D
*   Sınıf ağırlıklandırma (Class Weighting)
*   Threshold optimizasyonu ve Focal Loss adaptasyonu
*   Kapsamlı hiperparametre, overfitting ve veri sızıntısı analizleri

### 2. MONAI DenseNet
*   Deep Fine-Tuning süreçleri
*   Threshold optimizasyonu ve Focal Loss
*   Sınıf ağırlıklandırma

---

## 📊 DenseNet3D: Overfitting Analizi ve Kritik Noktalar

3 boyutlu MR hacimleri üzerinden anomali tespiti yapması planlanan DenseNet3D modeli, kurulan pipeline üzerinde başlangıç kapasitesini ölçmek amacıyla 20 epoch boyunca eğitilmiştir. 

Eğitim verileri incelendiğinde:
*   Modelin **6. epoch** civarında en dengeli (optimum) haline ulaştığı saptanmıştır.
*   **13. epoch'tan itibaren** eğitim hatası (train loss) düşmeye devam ederken, doğrulama hatasının (val loss) artmaya başlaması modelin **overfitting (ezberleme)** eğilimine girdiğini net bir şekilde göstermiştir.
*   Kurulan *Checkpoint* sistemi sayesinde modelin bozulmadan önceki en iyi ağırlıkları (weights) korumaya alınmıştır.

> **Not:** *Depodaki ilgili klasörlerde eğitim sürecini gösteren Loss ve Accuracy eğrilerinin grafiklerini (Şekil 1) bulabilirsiniz.*

## 🛡️ Veri Güvenliği ve Sızıntı (Leakage) Denetimi

Medikal yapay zeka modellerinde en kritik konulardan biri olan veri sızıntısı (Data Leakage) titizlikle kontrol edilmiştir.
*   **Train Seti:** 2082 hasta
*   **Validation Seti:** 445 hasta
*   **Sonuç:** Eğitim ve doğrulama setleri arasında yapılan çapraz sorguda **kesişim kümesi 0 (sıfır)** çıkmıştır. Modelin başarı oranları tamamen dürüst testlere dayanmaktadır.

## 🔬 Teknik Analizler (Ablasyon ve Hiperparametre)

Modelin farklı bileşenlerinin ve donanım sınırlarının performansa etkisini ölçmek için aşağıdaki analizler yapılmıştır:

| Analiz Türü | Varyasyon | Val Acc (%) | Gözlem ve Değerlendirme |
| :--- | :--- | :--- | :--- |
| **Ablasyon** | Baseline (Full) | **%52.58** | 3D Modelin mevcut (baseline) başlangıç performansıdır. |
| **Ablasyon** | Focal Loss Yok | **%43.20** | Sınıf dengesizliği modelin odağını ciddi ölçüde bozmuştur. |
| **Hiperparametre** | Batch Size: 8 | **N/A** | M4 Max GPU belleği 3D veri yükü (tensor boyutu) altında sınırlarına ulaşmış ve OOM (Out of Memory) vermiştir. |

---

*Mevcut durum itibarıyla projenin veri işleme hattı ve teknik altyapısı sağlam bir şekilde kurulmuş olup, model mimarisi ileri düzey geliştirmelere hazır hale getirilmiştir.*
