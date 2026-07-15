# Model Kartı (Model Card)

## Genel Bilgiler

| Alan | Değer |
|---|---|
| Model adı | `best_model_balanced_crop74.pth` |
| Mimari | ConvNeXt-Tiny (torchvision), ImageNet ön-eğitimli |
| Görev | İkili sınıflandırma — Diz MR: **Normal** vs **Abnormal** |
| Çerçeve | PyTorch + torchvision |
| Giriş biçimi | 2.5D — ardışık 3 kesit (gri tonlama → 3 kanal), 224×224 |
| Sınıf etiketleri | `0 = normal`, `1 = abnormal` |
| Sınıflandırma başı | `Dropout(0.5) + Linear(768, 2)` |
| Parametre sayısı | ~28M (ConvNeXt-Tiny) |
| Alan (domain) | 3D-DESS diz MR sekansı, 64 kesit/hacim |

## Amaçlanan Kullanım

- **Amaçlanan:** Diz MR hacimlerinde normal/abnormal ön-tarama üzerine akademik
  araştırma, yöntem geliştirme ve eğitim amaçlı gösterim (TÜBİTAK 2209-A).
- **Amaçlanmayan:** Klinik tanı, tedavi kararı veya tek başına tıbbi kullanım.
  Model bir karar-destek prototipidir, sertifikalı bir tıbbi cihaz değildir.

## Eğitim Verisi

- Kaynak: 3D-DESS diz MR hacimleri (her hacim 64 adet 256×256 PNG kesit).
- Bölünme (hacim/hasta düzeyinde, sızıntısız):

| Split | Normal | Abnormal | Toplam |
|---|---|---|---|
| train | 1161 | 921 | 2082 |
| val | 248 | 197 | 445 |
| test | 250 | 199 | 449 |

- Sınıf dengesi ≈ %56 normal / %44 abnormal (ılımlı dengesizlik).
- **Veri sızıntısı kontrolü:** Setler arasında kimlik (klasör adı) ve içerik
  (piksel hash) düzeyinde **örtüşme yoktur** (bkz. `sonuclar/veri_sizintisi_raporu.txt`).

## Eğitim Yöntemi (özet)

- **2 fazlı transfer öğrenme:** Faz 1 (10 epoch, yalnızca baş, AdamW lr=1e-3) →
  Faz 2 (≤40 epoch, tam ince ayar, AdamW lr=5e-5, CosineAnnealingLR, early-stop=8).
- **Kayıp:** Focal Loss (γ=2.0), sınıf ağırlığı `[1.0, 1.25]`.
- **Düzenlileştirme:** Dropout 0.5; augmentasyon (yatay çevirme, affine, color
  jitter, random erasing); AMP (karışık hassasiyet).
- Ayrıntı: `dokumanlar/metodoloji.md` ve `dokumanlar/hiperparametre_analizi.md`.

## Değerlendirme Sonuçları (Bağımsız Test Seti, n=449)

| Metrik | Değer |
|---|---|
| Doğruluk (Accuracy) | %72.4 |
| Duyarlılık (Sensitivity) | %65.3 |
| Özgüllük (Specificity) | %78.0 |
| ROC-AUC | 0.797 |
| PR-AUC | 0.780 |
| Makro-F1 | 0.718 |

**Sınıf bazlı:**

| Sınıf | Precision | Recall | F1 | Destek |
|---|---|---|---|---|
| Normal (Sağlıklı) | 0.739 | 0.780 | 0.759 | 250 |
| Abnormal (Hasta) | 0.703 | 0.653 | 0.677 | 199 |

**Karışıklık matrisi:** TN=195, FP=55, FN=69, TP=130.

İlgili grafikler: `sonuclar/karmasiklik_matrisi.png`, `roc_egrisi.png`, `pr_egrisi.png`.

## Açıklanabilirlik

İki tamamlayıcı XAI yöntemi kullanılır (bkz. `dokumanlar/aciklanabilirlik_xai.md`):
- **Grad-CAM** (`sonuclar/gradcam_ornekleri/`): modelin kararlarını verirken
  ağırlıklı olarak **diz ekleminin merkezî yapılarına** odaklandığını gösterir.
- **SHAP** (`sonuclar/shap_ornekleri/`): piksel düzeyinde imzalı katkı haritaları;
  Grad-CAM ile aynı anatomik bölgeleri işaret ederek bulguyu bağımsız doğrular.

Her iki yöntem de klinik olarak beklenen ilgi bölgesiyle tutarlıdır.

## Risk Skoru (0-100)

Model, `P(abnormal)` çıktısını 0-100 arası bir risk skoruna çevirir ve hacmi
**Düşük (0-33) / Orta (34-66) / Yüksek (67-100)** grubuna atar. Test setinde bu
gruplardaki gerçek hasta oranı **%13.6 → %43.2 → %88.9** (monoton) olup skorlamanın
ayrım gücünü doğrular (bkz. `dokumanlar/risk_skorlama.md`).

## Sınırlamalar ve Etik

- **Duyarlılık görece düşük (%65.3):** Standart eşikte (0.50) model bazı hasta
  hacimlerini kaçırır (69 yanlış negatif). **Çözüm — karar eşiği ayarı:** eşik
  0.43'e düşürülünce duyarlılık **%81.9**'a çıkar (kaçan hasta 36), özgüllük ise
  %58'e iner. Bu operasyon noktası val setinde seçilip test setinde doğrulanmıştır
  (bkz. `dokumanlar/esik_analizi.md`); arayüz ve CLI'da "Yüksek Duyarlılık" modu
  olarak sunulur.
- **Tek merkez / tek sekans (3D-DESS):** Farklı cihaz, protokol veya popülasyonda
  başarım düşebilir (dağılım kayması). Dış geçerleme yapılmamıştır.
- **2.5D yaklaşımı** yalnızca orta 3 kesiti kullanır; hacmin tamamındaki bilgi
  kısmen göz ardı edilir.
- **Hafif overfitting** (train−test GAP ≈ 9.9 puan) mevcuttur; daha fazla veri
  veya düzenlileştirme ile azaltılabilir.
- Model çıktıları klinik doğrulamadan geçmemiştir.

## Yeniden Üretilebilirlik

Tüm metrikler `kaynak_kod/degerlendirme.py` ile `data/test` üzerinde üretilebilir.
Ortam: Python 3.12, PyTorch (CUDA). Bkz. `requirements.txt`.
