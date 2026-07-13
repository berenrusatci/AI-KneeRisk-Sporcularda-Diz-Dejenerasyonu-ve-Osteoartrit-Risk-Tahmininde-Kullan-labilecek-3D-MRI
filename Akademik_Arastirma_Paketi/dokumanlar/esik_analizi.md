# Karar Eşiği (Threshold) Analizi — Duyarlılığı Artırma

Model varsayılan olarak **argmax** ile, yani `P(abnormal) ≥ 0.50` kuralıyla karar
verir. Bu eşikte model **özgüllüğe** (sağlıklıları doğru eleme) meyillidir ve
duyarlılık görece düşüktür (%65.3). Ancak bir **tarama/triyaj** aracında öncelik,
hasta hacimlerini kaçırmamaktır → yüksek **duyarlılık** (sensitivity). Bu doküman,
karar eşiğini düşürerek duyarlılığı nasıl artırdığımızı ve bunun özgüllükle
ödünleşimini raporlar. Üretim: `kaynak_kod/esik_analizi.py`.

## Yöntem (test sızıntısını önleyerek)

- Karar eşikleri **yalnızca doğrulama (val) setinde** seçilir.
- Seçilen eşiklerle nihai metrikler **bağımsız test setinde** raporlanır.
- Bu ayrım, eşiğin test verisine göre optimize edilmesini (iyimser yanlılık)
  engeller ve akademik olarak doğru operasyon noktası seçimini sağlar.

Eşik seçim kriterleri:
- **Youden-J:** `duyarlılık + özgüllük − 1` maksimize edilir (dengeli nokta).
- **Hedef duyarlılık (≥0.85 / ≥0.90):** val'de bu duyarlılığı sağlayan **en yüksek**
  eşik seçilir (böylece özgüllük mümkün olduğunca korunur).

## Sonuçlar (metrikler TEST setinde, n=449)

| Senaryo | Eşik | Doğruluk | Duyarlılık | Özgüllük | PPV | Makro-F1 | Kaçan hasta (FN) | Yanlış alarm (FP) |
|---|---|---|---|---|---|---|---|---|
| Standart (0.50) | 0.500 | %72.4 | %65.3 | %78.0 | 0.703 | 0.718 | 69 | 55 |
| Youden-J (val) | 0.517 | %71.9 | %59.3 | %82.0 | 0.724 | 0.708 | 81 | 45 |
| **Duyarlılık ≥ 0.85 (val) — ÖNERİLEN** | **0.426** | %68.8 | **%81.9** | %58.4 | 0.610 | 0.688 | **36** | 104 |
| Duyarlılık ≥ 0.90 (val) | 0.398 | %64.4 | **%86.9** | %46.4 | 0.564 | 0.638 | 26 | 134 |

İlgili grafik: `sonuclar/esik_analizi.png` (duyarlılık/özgüllük vs eşik eğrileri).

## Yorum

- **Önerilen eşik (0.426):** Duyarlılık **%65.3 → %81.9** yükselir; kaçan hasta
  sayısı **69 → 36**'ya (yarıya yakın) düşer. Bedeli, özgüllüğün %78 → %58'e
  inmesi ve yanlış alarmların (FP) 55 → 104'e çıkmasıdır. Tarama bağlamında bu,
  savunulabilir bir ödünleşimdir: bir tarama testinin kaçırdığı hasta (yanlış
  negatif), fazladan incelemeye yönlendirdiği sağlıklı kişiden (yanlış pozitif)
  klinik olarak daha maliyetlidir.
- **Youden-J** bu modelde 0.50'nin hafif üzerinde (0.517) çıkar ve duyarlılığı
  *azaltır*; dolayısıyla "hasta kaçırmama" hedefi için uygun değildir. Dengeli
  doğruluk isteyen senaryolar için referans olarak tabloda tutulmuştur.
- **Daha agresif** tarama için 0.398 eşiği duyarlılığı %86.9'a çıkarır (kaçan
  hasta yalnızca 26) ancak özgüllük %46.4'e düşer.

## Kullanım

Önerilen eşik `sonuclar/esik_ayari.json` dosyasına yazılır; arayüz ve CLI bunu okur.

- **Streamlit arayüzü:** Yan paneldeki **"Karar Eşiği"** bölümünden
  *Yüksek Duyarlılık (0.43) — önerilen* modu seçilir (veya *Özel* kaydırıcı ile
  herhangi bir eşik denenir). Sonuç panelinde kullanılan eşik ve gerekirse
  "standart eşikte normal sayılırdı" uyarısı gösterilir.
- **Komut satırı:**
  ```
  python tahmin.py "<hacim>" --yuksek-duyarlilik   # önerilen eşik (0.43)
  python tahmin.py "<hacim>" --esik 0.40           # özel eşik
  ```

## Not

Karar eşiği modelin **ağırlıklarını değiştirmez**; yalnızca olasılık çıktısına
uygulanan operasyon noktasını kaydırır. Bu nedenle eşik, klinik önceliğe göre
(duyarlılık mı özgüllük mü) dağıtım anında serbestçe ayarlanabilir. Uygun eşik,
hedef kullanım senaryosunun yanlış-negatif / yanlış-pozitif maliyet dengesine göre
seçilmelidir.
