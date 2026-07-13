# Diz MR Sınıflandırma — Akademik Araştırma Paketi

**TÜBİTAK 2209-A · Spor ve Sağlık · Diz MR Görüntülerinde Normal / Abnormal Ayrımı**

Bu paket, eğitilmiş **ConvNeXt-Tiny (2.5D)** diz MR sınıflandırma modelinin
(`best_model_balanced_crop74.pth`) akademik düzeyde değerlendirilmesi, analizi ve
çalıştırılabilir bir demo arayüzüyle sunulması için hazırlanmıştır. Tüm sayısal
sonuçlar, `data/test` bağımsız test seti üzerinde bu paketteki scriptler
çalıştırılarak üretilmiştir (uydurma değer yoktur).

---

## 🔑 Öne Çıkan Sonuçlar (Bağımsız Test Seti, n=449)

| Metrik | Değer |
|---|---|
| Doğruluk (Accuracy) | **%72.4** |
| Duyarlılık (Sensitivity, hasta yakalama) | %65.3 |
| Özgüllük (Specificity, sağlıklı eleme) | %78.0 |
| ROC-AUC | **0.797** |
| PR-AUC (Average Precision) | 0.780 |
| Makro-F1 | 0.718 |
| Genelleme Boşluğu (GAP, train−test) | 9.9 puan (hafif overfitting) |

> **Yüksek Duyarlılık Modu:** Karar eşiği 0.50 → **0.43**'e düşürüldüğünde
> duyarlılık **%65.3 → %81.9**'a çıkar (kaçan hasta 69 → 36), özgüllük %78 → %58.
> Ayrıntı: `dokumanlar/esik_analizi.md`. Arayüzde yan panelden veya CLI'da
> `--yuksek-duyarlilik` ile etkinleştirilir.

- **Veri sızıntısı:** train/val/test setleri arasında hem kimlik hem piksel
  düzeyinde **hiçbir örtüşme yok** → sonuçlar güvenilir.
- **Ablasyon:** ConvNeXt-Tiny + Focal Loss + sınıf dengesi + CenterCrop,
  karşılaştırılan tüm mimarileri (EfficientNet-B0, ResNet50, önceki ConvNeXt)
  geride bırakmıştır.

---

## 📁 Klasör Yapısı

```
Akademik_Arastirma_Paketi/
├── README.md                     # Bu dosya
├── requirements.txt              # Bağımlılıklar
├── kaynak_kod/                   # Tüm Python scriptleri
│   ├── ortak.py                  # Merkezi modül (model, veri seti, ön işleme) — TEK KAYNAK
│   ├── degerlendirme.py          # Ana test değerlendirmesi (metrikler, ROC/AUC, PR, CM)
│   ├── gradcam.py                # Grad-CAM açıklanabilirlik ısı haritaları
│   ├── model_karsilastirma.py    # Mimari ablasyon / model karşılaştırması
│   ├── gap_overfitting_analizi.py# GAP (genelleme boşluğu) ve overfitting analizi
│   ├── veri_sizintisi_kontrolu.py# Veri sızıntısı (data leakage) kontrolü
│   ├── hiperparametre_analizi.py # Hiperparametre analizi (+ ampirik tarama)
│   ├── esik_analizi.py           # Karar eşiği ayarı — duyarlılık artırma
│   ├── derinlik_analizi.py       # Kayan pencere + birleştirme (tek konfig ablasyon)
│   ├── derinlik_tarama.py        # Kayan pencere konfig taraması (merkez×step×k)
│   ├── tahmin.py                 # Komut satırı tek hacim çıkarımı (eşik destekli)
│   └── arayuz.py                 # Streamlit demo arayüzü (eşik + kesit modlu)
├── sonuclar/                     # Üretilen tüm çıktılar (metrikler, grafikler)
│   ├── metrikler.json / siniflandirma_raporu.txt
│   ├── karmasiklik_matrisi.png / roc_egrisi.png / pr_egrisi.png
│   ├── model_karsilastirma.csv / .png
│   ├── gap_overfitting.json / .png / ogrenme_egrileri.png
│   ├── hiperparametre_analizi.json / hiperparametre_tarama.png
│   ├── esik_analizi.json / .csv / .png / esik_ayari.json
│   ├── veri_sizintisi_raporu.txt
│   └── gradcam_ornekleri/*.png
└── dokumanlar/                   # Akademik dokümanlar
    ├── model_karti.md
    ├── metodoloji.md
    ├── ablasyon_analizi.md
    ├── hiperparametre_analizi.md
    ├── esik_analizi.md
    └── gap_analizi.md
```

---

## ⚙️ Kurulum

Proje, `C:\Projelerim\Tubitak-Proje\.venv312` (Python 3.12, CUDA) ortamıyla
doğrulanmıştır. Bu ortam hazırdır; eksik paketler kurulmuştur.

Sıfırdan kurulum için:

```powershell
python -m venv .venv312
.\.venv312\Scripts\Activate.ps1
pip install -r requirements.txt
# GPU için torch/torchvision'ı resmi index ile kurun (requirements.txt notuna bakın)
```

Doğrulama:
```powershell
& "C:\Projelerim\Tubitak-Proje\.venv312\Scripts\python.exe" -c "import torch; print(torch.cuda.is_available())"
```

---

## ▶️ Çalıştırma

Aşağıdaki komutlar `kaynak_kod/` klasöründen çalıştırılır. `PY` kısaltması:
`& "C:\Projelerim\Tubitak-Proje\.venv312\Scripts\python.exe"`

| Amaç | Komut |
|---|---|
| Ana değerlendirme (metrikler + grafikler) | `PY degerlendirme.py` |
| Veri sızıntısı kontrolü | `PY veri_sizintisi_kontrolu.py` |
| GAP / overfitting analizi | `PY gap_overfitting_analizi.py` |
| Model karşılaştırma (ablasyon) | `PY model_karsilastirma.py` |
| Grad-CAM örnekleri | `PY gradcam.py --ornek-sayisi 3` |
| Hiperparametre analizi (+ ampirik tarama) | `PY hiperparametre_analizi.py --tara` |
| **Karar eşiği analizi (duyarlılık artırma)** | `PY esik_analizi.py` |
| Tek hacim tahmini (CLI) | `PY tahmin.py "<hacim_klasörü_yolu>"` |
| Tek hacim, yüksek duyarlılık | `PY tahmin.py "<hacim>" --yuksek-duyarlilik` |

### Demo Arayüzü (Streamlit)

```powershell
cd kaynak_kod
& "C:\Projelerim\Tubitak-Proje\.venv312\Scripts\streamlit.exe" run arayuz.py
```

Tarayıcıda `http://localhost:8501` açılır. Test setinden hacim seçebilir veya
kendi PNG kesitlerinizi yükleyebilirsiniz; arayüz tahmini, sınıf olasılıklarını,
kullanılan 3 kesiti ve Grad-CAM ısı haritasını gösterir.

---

## 🧠 Model Özeti

- **Mimari:** ConvNeXt-Tiny (ImageNet ön-eğitimli), sınıflandırma başı
  `Dropout(0.5) + Linear(768, 2)`.
- **Giriş (2.5D):** Bir MR hacminin ortasındaki 3 kesit gri tonlamalı olarak 3
  kanala yığılır → `[3, 224, 224]`.
- **Ön işleme (birebir korunur):** `CenterCrop(280) → Resize(224) → Normalize(ImageNet)`.
- **Sınıflar:** `0 = normal`, `1 = abnormal`.

Ayrıntılar için `dokumanlar/model_karti.md` ve `dokumanlar/metodoloji.md`.

---

## ⚠️ Sorumluluk Reddi

Bu paket **akademik/araştırma** amaçlıdır. Model bir **klinik tanı aracı
değildir** ve tıbbi karar için tek başına kullanılamaz.
