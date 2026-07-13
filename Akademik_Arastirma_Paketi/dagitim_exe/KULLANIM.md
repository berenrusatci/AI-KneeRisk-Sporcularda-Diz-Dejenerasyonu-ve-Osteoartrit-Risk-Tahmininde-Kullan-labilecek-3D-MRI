# Diz MR Sınıflandırma — Çalıştırılabilir Sürüm (DizMR.exe)

ConvNeXt-Tiny 2.5D diz MR sınıflandırma modelinin (Normal / Abnormal)
kurulum gerektirmeyen Windows sürümü. Python, PyTorch vb. kurmanıza
gerek yoktur; her şey bu klasörün içindedir.

## Çalıştırma

1. `DizMR.exe` dosyasına çift tıklayın.
2. Herhangi bir pencere AÇILMAZ; 10–30 saniye içinde arayüz
   tarayıcınızda **kendiliğinden** açılır (http://localhost:8501).
3. Kapatmak için arayüzün yan panelindeki **🛑 Uygulamayı Kapat**
   düğmesini kullanın.

> İlk açılış, dosyaların belleğe yüklenmesi ve virüs taraması nedeniyle
> yavaş olabilir (1 dakikaya kadar); bekleyin, tekrar tekrar tıklamayın.
> Uygulama zaten çalışıyorsa çift tıklamak yeni kopya açmaz, mevcut
> arayüzü tarayıcıda tekrar açar. Model CPU üzerinde çalışır.

## Arayüz

- **Test setinden seç:** `data/test/` içindeki örnek hacimlerden birini
  seçip "Tahmin Et" deyin. Gerçek etiketle karşılaştırma gösterilir.
- **Kendi kesitlerini yükle:** En az 3 ardışık PNG kesit yükleyin;
  model ortadaki 3 kesiti kullanır.
- Sonuç ekranında sınıf olasılıkları, modele giren 3 kesit ve
  Grad-CAM ısı haritası (modelin odaklandığı bölge) gösterilir.
- Yan panelden karar eşiği (standart 0.50 / yüksek duyarlılık / özel)
  seçilebilir.
- **Kesit Modu** (yan panel): *Orta 3 kesit* (temel) ya da *Merkezi kayan
  pencere* (önerilen). İkincisi, hacmin ortadaki bölgesini 3'erli
  pencerelerle tarayıp skorları ortalar; test setinde temele göre biraz
  daha yüksek doğruluk/ROC-AUC verir (bkz. `dokumanlar/derinlik_analizi.md`).

## Klasör düzeni

| Klasör | İçerik |
|---|---|
| `_internal/` | Uygulamanın çalışma dosyaları — **dokunmayın** |
| `uygulama/` | Arayüz ve model kaynak kodu (arayuz.py, ortak.py, gradcam.py) |
| `kaydedilen_modeller/` | Eğitilmiş model ağırlıkları (.pth) |
| `sonuclar/` | Test seti metrikleri ve önerilen eşik (JSON) |
| `data/test/` | Demo için örnek test hacimleri (her sınıftan birkaç adet) |

Daha fazla örnek hacim eklemek için, klasör başına en az 3 PNG kesit
içerecek şekilde `data/test/normal/` veya `data/test/abnormal/` altına
hacim klasörleri kopyalamanız yeterlidir.

## Sorun giderme

- **Günlükler:** Her başlatma ve olası hata şu iki dosyaya yazılır:
  `dizmr_gunluk.txt` (exe'nin yanında) ve
  `%LOCALAPPDATA%\DizMR\dizmr_gunluk.txt`. Ölümcül hatalar ayrıca
  Windows iletişim kutusuyla gösterilir.
- **Öz-kontrol:** `DizMR.exe --selftest` model yükleme, örnek tahmin ve
  Grad-CAM adımlarını arayüz açmadan test eder; sonucu iletişim
  kutusunda ve günlükte bildirir.
- **Tarayıcı açılmadıysa:** Tarayıcınızdan http://localhost:8501
  adresine kendiniz gidin.
- **Uygulama arka planda kaldıysa:** Görev Yöneticisi'nden DizMR.exe
  sonlandırılabilir (normal kapatma yolu arayüzdeki düğmedir).
- **Ağ hatası (WinError 10106):** Yönetici PowerShell'de
  `netsh winsock reset` çalıştırıp bilgisayarı yeniden başlatın.
- **Güvenlik uyarısı:** Windows SmartScreen ilk çalıştırmada uyarı
  verebilir ("Yine de çalıştır" ile devam edin) — uygulama imzasızdır.

## Uyarı

⚠️ Bu araç **akademik/araştırma amaçlıdır**, klinik tanı aracı değildir.
(TÜBİTAK 2209-A öğrenci projesi kapsamında geliştirilmiştir.)
