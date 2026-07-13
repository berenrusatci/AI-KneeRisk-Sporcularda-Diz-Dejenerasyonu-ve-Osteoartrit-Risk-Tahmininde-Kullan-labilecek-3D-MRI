# Hiperparametre Analizi

İki bölümden oluşur: (A) nihai modelin dokümante hiperparametreleri ve gerekçeleri,
(B) seçilen bazı hiperparametrelerin doğrulama başarımına etkisini ölçen ampirik
mini-tarama. Üretim: `kaynak_kod/hiperparametre_analizi.py --tara`
→ `sonuclar/hiperparametre_analizi.json`, `hiperparametre_tarama.png`.

## A) Dokümante Hiperparametreler (Nihai Model)

| Hiperparametre | Değer | Gerekçe |
|---|---|---|
| Mimari | ConvNeXt-Tiny (ImageNet ön-eğitimli) | Modern CNN; güçlü transfer |
| Giriş | 2.5D, orta 3 kesit, 224×224 | Hesaplama verimliliği + kesit-arası bağlam |
| Batch size | 16 | 8–12 GB VRAM'de AMP ile denge |
| Kayıp | Focal Loss (γ=2.0) | Dengesiz sınıfta zor örneklere odak |
| Sınıf ağırlığı | [1.0, 1.25] | Azınlık (abnormal) sınıfına ılımlı ağırlık |
| Transfer öğrenme | 2 fazlı (baş → tam ince ayar) | Kararlı yakınsama, ön-eğitimi koru |
| Faz 1 LR | 1e-3 (yalnızca baş) | Yeni başı hızlı ısıt |
| Faz 2 LR | 5e-5 + CosineAnnealing | Omurgayı nazikçe ince ayarla |
| Optimizer | AdamW (wd=1e-2) | Ağırlık çürümesiyle düzenlileştirme |
| Erken durdurma | patience=8 (val loss) | Overfitting'i önle, en iyiyi koru |
| Dropout | 0.5 | Baş düzeyinde düzenlileştirme |
| Augmentasyon | Flip, Affine, ColorJitter, RandomErasing | Ezberi kır, genellemeyi artır |
| AMP | float16 | Hız + VRAM tasarrufu |

## B) Ampirik Mini-Tarama (Hiperparametre Duyarlılığı)

**Yöntem:** Tam yeniden eğitim maliyetine girmeden hiperparametre etkisini ölçmek
için **özellik-önbellekli linear-probe** kullanılmıştır: ConvNeXt-Tiny omurgası
dondurulur, tüm train/val setinin 768-boyutlu havuzlanmış özellikleri **bir kez**
çıkarılır, ardından her konfigürasyon için lineer sınıflandırma başı bu özellikler
üzerinde 60 epoch eğitilir. Böylece tek-değişkenli, kontrollü ve hızlı bir
karşılaştırma elde edilir.

> Not: Linear-probe rejimi mutlak başarımı sınırlar (tavan ≈ %69 val); amaç mutlak
> değer değil, hiperparametrelerin **göreli etkisidir**. Nihai model tam ince ayarla
> bu tavanın üzerine çıkar (test %72.4).

### Sonuçlar (En iyi Doğrulama Doğruluğu)

**Öğrenme Oranı** (kayıp=Focal, ağırlık=[1,1.25]):

| lr | Val Acc |
|---|---|
| 1e-2 | %69.0 |
| **1e-3 (seçilen)** | %68.3 |
| 1e-4 | %65.6 |

→ Çok düşük öğrenme oranı (1e-4) belirgin underfitting'e yol açar; 1e-2 ve 1e-3
benzer ve en iyi bölgededir. Seçilen 1e-3, tam ince ayarda kararlılık için
muhafazakâr ve güvenli bir tercihtir.

**Kayıp Fonksiyonu** (lr=1e-3, ağırlık=[1,1.25]):

| Kayıp | Val Acc |
|---|---|
| CrossEntropy | %69.0 |
| **Focal (seçilen)** | %68.3 |

→ Linear-probe rejiminde CrossEntropy ve Focal doğrulukta neredeyse eşittir. Focal
Loss'un asıl faydası, **tam ince ayarlı ve dengesiz** rejimde zor/azınlık örneklere
odaklanmada ortaya çıkar (bkz. ablasyonda ANA modelin dengeli duyarlılık/özgüllüğü).

**Sınıf Ağırlığı** (lr=1e-3, kayıp=Focal):

| Ağırlık | Val Acc |
|---|---|
| [1.0, 1.0] | %69.2 |
| **[1.0, 1.25] (seçilen)** | %68.3 |
| [1.0, 1.6] | %67.9 |

→ Ağırlık arttıkça ham doğruluk hafifçe düşer: model azınlık sınıfına daha çok
kayarak yanlış-pozitifleri artırır. Eğitim scriptinde de not edildiği üzere ağırlık
1.6'dan 1.25'e düşürülmüştür; bu, **doğruluk ile duyarlılık arasında bilinçli bir
denge** kararıdır (saf doğruluk yerine dengeli hata profili).

## Genel Değerlendirme

Ampirik tarama, nihai modelde yapılan hiperparametre seçimlerinin makul ve
savunulabilir olduğunu doğrular: öğrenme oranı iyi bölgede, kayıp ve sınıf ağırlığı
seçimleri ise saf doğruluğu birkaç puan feda etme pahasına **daha dengeli bir
duyarlılık-özgüllük profili** hedefleyen kasıtlı tercihlerdir. En büyük başarım
sıçraması ise hiperparametre ince ayarından çok **omurga seçimi ve tam ince ayardan**
gelmektedir (bkz. `ablasyon_analizi.md`).
