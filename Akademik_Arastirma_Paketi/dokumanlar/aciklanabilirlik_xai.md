# Açıklanabilir Yapay Zeka (XAI): Grad-CAM ve SHAP

Başvuru **Hedef 4**, modelin kararlarını iki tamamlayıcı XAI yöntemiyle
açıklamayı taahhüt eder: **Grad-CAM** ve **SHAP**. Bu doküman her ikisinin
uygulamasını ve bulgularını özetler. Üretim: `kaynak_kod/gradcam.py` ve
`kaynak_kod/shap_analizi.py`.

## 1. Grad-CAM (Gradient-weighted Class Activation Mapping)

- **Yöntem:** Modelin son özellik bloğuna (`features[-1]`, ConvNeXt son evre)
  forward/backward hook takılır; hedef sınıf skorunun bu bloğa göre gradyanlarının
  uzamsal ortalaması kanal ağırlığı olarak kullanılıp aktivasyonlarla çarpılır.
  Ekstra kütüphane gerektirmez.
- **Çıktı:** `sonuclar/gradcam_ornekleri/*.png` — orta kesit üzerine kırmızı
  (yüksek etkili) ısı haritası bindirmesi.
- **Bulgu:** Modelin kararları ağırlıklı olarak **diz ekleminin merkezî
  yapılarına** (eklem aralığı, kıkırdak/kemik kenarları) dayanır; bu, klinik olarak
  beklenen ilgi bölgesiyle tutarlıdır.
- **Kullanım:** `python gradcam.py --ornek-sayisi 3`

## 2. SHAP (SHapley Additive exPlanations)

- **Yöntem:** `shap.GradientExplainer` ile, bir arka plan (background) örneklem
  kümesine göre **beklenen gradyanlar (expected gradients)** üzerinden her pikselin
  "abnormal" kararına katkısı (Shapley değeri) hesaplanır. 2.5D girdinin 3 kanalı
  (komşu kesitler) toplanarak 224×224 imzalı bir katkı haritası elde edilir.
  Kaynak: Lundberg & Lee (2017).
- **Çıktı:** `sonuclar/shap_ornekleri/*.png` — diverging (bwr) renk skalası:
  - 🔴 Kırmızı (+): abnormal olasılığını **artıran** bölgeler,
  - 🔵 Mavi (−): abnormal olasılığını **azaltan** bölgeler.
- **Bulgu:** SHAP katkıları da eklem çizgisi ve kemik/kıkırdak sınırları boyunca
  yoğunlaşır; Grad-CAM'in gösterdiği ilgi bölgeleriyle **niteliksel olarak
  tutarlıdır**, ancak piksel düzeyinde işaretli (artıran/azaltan) daha ince bir
  ayrıştırma sunar.
- **Kullanım:** `python shap_analizi.py --ornek-sayisi 3 --arka-plan 24`

## Grad-CAM vs SHAP — Neden İkisi Birden?

| Özellik | Grad-CAM | SHAP (GradientExplainer) |
|---|---|---|
| Çözünürlük | Kaba (son evre 7×7 → yükseltilmiş) | Piksel düzeyi (ince) |
| İşaret | Yalnızca pozitif (odak) | İmzalı (+/−: artıran/azaltan) |
| Teorik temel | Gradyan-aktivasyon | Oyun-teorik (Shapley), toplanabilir |
| Referans | Selvaraju et al. (2017) | Lundberg & Lee (2017) |

İki yöntem birbirini tamamlar: Grad-CAM "nereye baktı"yı hızlı ve kaba gösterirken,
SHAP her pikselin katkısını işaretli ve teorik olarak tutarlı biçimde niceler. İki
bağımsız yöntemin **aynı anatomik bölgeleri** işaret etmesi, modelin kararlarının
klinik güvenilirliğini artırır.

## Bağımlılık

SHAP için `shap>=0.44` gereklidir (`requirements.txt`'e eklendi). Grad-CAM ek
bağımlılık gerektirmez.
