# dagitim_exe — DizMR.exe Derleme Klasörü

Akademik paketteki Streamlit arayüzünü (`kaynak_kod/arayuz.py`) ve modeli,
kurulum gerektirmeyen tek klasörlük bir Windows uygulamasına (**DizMR.exe**)
çeviren derleme dosyaları.

## Dosyalar

| Dosya | Görev |
|---|---|
| `launcher.py` | Exe giriş noktası: `DIZMR_KOK` ortam değişkenini ayarlar, Streamlit'i başlatır, sunucu hazır olunca tarayıcıyı kendisi açar. Tek kopya koruması vardır (çalışan kopya varsa tarayıcıyı ona yönlendirir). `--selftest` bayrağı ile arayüz açmadan model + tahmin + Grad-CAM testi yapar. |
| `DizMR.spec` | PyInstaller derleme tanımı (onedir, **konsolsuz/windowed**). |
| `derle.ps1` | Tek komutla derleme: PyInstaller'ı çalıştırır, sonra modeli, kaynak kopyalarını, metrik JSON'larını ve örnek test hacimlerini `dist/DizMR/` içine kopyalar. |
| `KULLANIM.md` | Son kullanıcı için kullanım kılavuzu (dist içine kopyalanır). |

## Derleme ortamı (bir kez kurulur)

Dağıtım boyutunu küçük tutmak için CUDA'lı `.venv312` DEĞİL, CPU-only torch
içeren ayrı bir ortam kullanılır (tek hacim çıkarımı CPU'da ~1 sn):

```powershell
& "C:\Users\kullanici-adi\AppData\Local\Programs\Python\Python312\python.exe" -m venv C:\Projelerim\Tubitak-Proje\.venv_exe
& C:\Projelerim\Tubitak-Proje\.venv_exe\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& C:\Projelerim\Tubitak-Proje\.venv_exe\Scripts\python.exe -m pip install streamlit matplotlib pillow numpy pyinstaller
```

## Derleme

```powershell
cd .\Akademik_Arastirma_Paketi\dagitim_exe
powershell -ExecutionPolicy Bypass -File .\derle.ps1
```

Çıktı: `dist\DizMR\` — bu klasörün tamamı dağıtılır (zip'lenip
paylaşılabilir). Test için: `dist\DizMR\DizMR.exe --selftest`

## Nasıl çalışıyor?

- `ortak.py`, `DIZMR_KOK` ortam değişkeni tanımlıysa tüm yolları
  (model, sonuçlar, data) o klasöre göre çözer; launcher bu değişkeni
  exe'nin klasörüne ayarlar. Değişken yoksa davranış eskisi gibidir,
  akademik paket etkilenmez.
- Exe **konsolsuz** derlenir: çift tıklamada pencere açılmaz, sunucu
  hazır olunca tarayıcı otomatik açılır. Konsol olmadığı için kapatma,
  arayüz yan panelindeki "🛑 Uygulamayı Kapat" düğmesiyle yapılır
  (`arayuz.py` bu düğmeyi yalnızca `DIZMR_KOK` tanımlıyken gösterir).
  Konsollu ilk sürümde açılışın ilk saniyeleri boş siyah pencereydi ve
  bu aşamada pencerenin kapatılması uygulamayı iz bırakmadan
  öldürüyordu; konsolsuz tasarım bu kırılganlığı ortadan kaldırır.
- Tanı için her başlatma `dizmr_gunluk.txt` (exe'nin yanı) ve
  `%LOCALAPPDATA%\DizMR\dizmr_gunluk.txt` dosyalarına yazılır; konsolsuz
  modda stdout/stderr `dizmr_cikti.txt`'ye bağlanır, ölümcül hatalar
  Windows iletişim kutusuyla gösterilir.
- `arayuz.py` / `ortak.py` / `gradcam.py` exe'ye gömülmez; dist içindeki
  `uygulama/` klasöründen çalışır. Böylece dağıtımda küçük kod
  düzeltmeleri yeniden derleme gerektirmez ve kod şeffaf kalır.
- Ön işleme zinciri (CenterCrop(280) → Resize(224) → ImageNet normalize)
  birebir `ortak.py`'den geldiği için exe ile paket aynı sonucu üretir.

> Not: `build/` ve `dist/` klasörleri derleme çıktısıdır; sürüm
> kontrolüne/yedeğe eklenmesi gerekmez, `derle.ps1` yeniden üretir.
