# GAP (Genelleme Boşluğu), Overfitting ve Veri Sızıntısı Analizi

Bu doküman modelin **ezberleme (overfitting)** eğilimini, **genelleme boşluğunu (GAP)**
ve **veri sızıntısı** durumunu birlikte değerlendirir. Üretim:
`kaynak_kod/gap_overfitting_analizi.py` (→ `sonuclar/gap_overfitting.json`, `.png`,
`ogrenme_egrileri.png`) ve `kaynak_kod/veri_sizintisi_kontrolu.py`
(→ `sonuclar/veri_sizintisi_raporu.txt`).

## 1. Split Bazlı Başarım

Ana model (`best_model_balanced_crop74.pth`) **aynı deterministik ön işlemeyle**
(augmentasyonsuz) train/val/test üçlüsünde değerlendirilmiştir. Augmentasyonsuz
"temiz" eğitim başarımı, ezber ölçümü için doğru referanstır.

| Split | n | Doğruluk | Makro-F1 | ROC-AUC |
|---|---|---|---|---|
| Eğitim (train) | 2082 | %82.3 | 0.821 | 0.914 |
| Doğrulama (val) | 445 | %74.8 | 0.746 | 0.826 |
| Test | 449 | %72.4 | 0.718 | 0.797 |

## 2. Genelleme Boşluğu (GAP)

| Boşluk | Değer |
|---|---|
| GAP Doğruluk (train − test) | **9.9 puan** |
| GAP Doğruluk (train − val) | 7.5 puan |
| GAP ROC-AUC (train − test) | 0.117 |
| GAP Makro-F1 (train − test) | 0.103 |

**Yorum:** GAP ≈ %9.9, "küçük–orta" aralıktadır ve **hafif overfitting**e işaret
eder. Bu düzey, tıbbi görüntüleme literatüründe akademik olarak **kabul edilebilir**
kabul edilir: model eğitim verisini kısmen daha iyi tanımakta ancak test setine
makul biçimde genellemektedir (test AUC 0.797). Önemli olarak, val (%74.8) ve test
(%72.4) başarımlarının **birbirine yakın** olması, val setinin test için güvenilir
bir vekil olduğunu ve model seçiminin sağlıklı yapıldığını gösterir.

## 3. Öğrenme Eğrileri

Eğitim sürecine ait kayıp/doğruluk eğrileri `sonuclar/ogrenme_egrileri.png`
dosyasındadır (Faz 1 → Faz 2 geçişi kesikli çizgiyle işaretlidir). Eğrilerde:
- Eğitim ve doğrulama kayıplarının birlikte düşmesi sağlıklı öğrenmeye,
- Erken durdurmanın en iyi doğrulama noktasında checkpoint almasına
işaret eder. Doğrulama kaybının eğitim kaybından belirgin biçimde ayrışıp
yükselmemesi, ağır bir overfitting olmadığını destekler.

## 4. Overfitting'i Sınırlayan Tasarım Kararları

Modelde ezberi baskılamak için birden çok mekanizma kullanılmıştır:
- **Dropout 0.5** (sınıflandırma başında),
- **Veri augmentasyonu** (flip, affine, color jitter, random erasing),
- **AdamW ağırlık çürümesi** (wd=1e-2),
- **Erken durdurma** (patience=8, val loss üzerinden),
- **Transfer öğrenme** (ImageNet ön-eğitimi, sıfırdan eğitime göre daha az ezber).

## 5. Veri Sızıntısı (Data Leakage) Kontrolü

Genelleme boşluğunun anlamlı olabilmesi için test setinin eğitimden tamamen bağımsız
olması şarttır. İki bağımsız yöntemle doğrulanmıştır:

1. **Kimlik (klasör adı) bazlı:** train/val/test hacim kimlikleri arasında kesişim
   **YOK** (indeksler sınıf içinde global ve ayrıktır).
2. **İçerik (piksel hash) bazlı:** Temsili kesitlerin MD5 özetleri üzerinden setler
   arası özdeş içerik **YOK**; split-içi tekrarlanan içerik de **YOK**.

**Sonuç:** `train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅` (hem kimlik hem
piksel düzeyinde). Yani test başarımı gerçek, sızıntısız bir genelleme ölçümüdür ve
yukarıdaki GAP değerleri güvenilirdir.

## 6. Özet

- Model **hafif** overfitting göstermektedir (GAP ≈ 9.9 puan), bu akademik olarak
  kabul edilebilir düzeydedir.
- val/test yakınlığı ve sızıntısızlık, sonuçların **güvenilir** olduğunu gösterir.
- İyileştirme yönleri: daha fazla/çeşitli veri, daha güçlü düzenlileştirme veya
  hacmin tamamını kullanan 3D/dizi tabanlı modeller ile GAP daha da azaltılabilir.
