# Risk Skorlama Sistemi (0-100) ve Risk Grupları

Bu doküman, başvuru **Hedef 5**'in uygulamasını raporlar: modelin `P(abnormal)`
olasılığını **0-100 arası bir diz dejenerasyonu / osteoartrit risk skoruna**
dönüştürmek ve her hacmi **Düşük / Orta / Yüksek** risk grubuna atamak. Üretim:
`kaynak_kod/risk_skorlama.py` → `sonuclar/risk_skorlama.{json,csv,png}`. Skorlama
mantığı merkezi modülde tanımlıdır (`ortak.risk_skoru`, `ortak.risk_grubu`) ve
CLI (`tahmin.py`) ile arayüzde (`arayuz.py`) doğrudan gösterilir.

## Tanım

- **Risk skoru** = `round(P(abnormal) × 100)` → 0-100 arası tamsayı.
- **Risk grupları:**

  | Grup | Skor aralığı |
  |---|---|
  | 🟢 Düşük | 0 – 33 |
  | 🟡 Orta | 34 – 66 |
  | 🔴 Yüksek | 67 – 100 |

Skor, modelin kalibre edilmemiş olasılığının doğrudan yüzdesidir; gruplar eşit
üçe bölmedir. Karar eşiği (bkz. `esik_analizi.md`) ikili sınıflandırma için ayrı
bir operasyon noktasıyken, risk skoru **sürekli bir sıralama/triyaj** çıktısıdır.

## Doğrulama (Bağımsız Test Seti, n=449)

Skorlama sisteminin ayrım gücü, her risk grubundaki **gerçek** hasta (abnormal)
oranıyla test edilir. Geçerli bir skorlamada bu oran Düşük → Orta → Yüksek
doğrultusunda **monoton artmalıdır**:

| Risk Grubu | Adet | Gerçek Hasta | Hasta Oranı | Ortalama Skor |
|---|---|---|---|---|
| 🟢 Düşük | 66 | 9 | **%13.6** | 29.8 |
| 🟡 Orta | 329 | 142 | **%43.2** | 47.8 |
| 🔴 Yüksek | 54 | 48 | **%88.9** | 76.2 |

- **Monotonluk:** ✅ Geçerli (%13.6 < %43.2 < %88.9).
- **Skor–etiket korelasyonu (Pearson r):** 0.519.

**Yorum:** Yüksek risk grubuna atanan hacimlerin **%88.9'u** gerçekten abnormaldir;
Düşük risk grubunda bu oran yalnızca **%13.6**'dır. Yani risk skoru, olasılık
eşiğinden bağımsız olarak klinik önceliklendirme için anlamlı bir sıralama sağlar:
yüksek skorlu hacimler öncelikli incelemeye yönlendirilebilir.

## Kullanım

- **CLI:** `python tahmin.py "<hacim>"` çıktısında `Risk Skoru: X/100 -> <grup> risk grubu`.
- **Arayüz:** Tahmin sonucu panelinde "🧮 Dejenerasyon/OA Risk Skoru" metriği ve
  renkli risk grubu rozeti gösterilir.
- **Toplu doğrulama:** `python risk_skorlama.py` — grup özetleri, monotonluk ve
  skor dağılım grafiği (`sonuclar/risk_skorlama.png`).

## Sınırlama

Skor, kalibre edilmemiş softmax olasılığına dayanır; mutlak bir olasılık
kalibrasyonu (ör. Platt/İzotonik) uygulanmamıştır. Gruplar arası **göreli**
sıralama geçerlidir, ancak "risk skoru 70 = %70 hastalık olasılığı" biçiminde
mutlak bir yorum garanti edilmez. Kalibrasyon, gelecek çalışma olarak önerilir.
