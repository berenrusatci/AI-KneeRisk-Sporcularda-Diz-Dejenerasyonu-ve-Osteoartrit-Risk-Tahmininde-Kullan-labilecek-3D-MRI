# -*- coding: utf-8 -*-
r"""
launcher.py — DizMR.exe Giriş Noktası (konsolsuz / windowed)
============================================================

PyInstaller ile paketlenen Streamlit arayüzünü başlatır. Exe konsolsuz
derlendiği için kullanıcı yalnızca tarayıcıda açılan arayüzü görür;
tüm tanı bilgisi günlük dosyalarına yazılır, ölümcül hatalar Windows
iletişim kutusuyla gösterilir.

Dağıtım klasörü düzeni (exe'nin yanında):
    DizMR.exe
    _internal/               <- PyInstaller çalışma dosyaları (dokunmayın)
    uygulama/                <- arayuz.py, ortak.py, gradcam.py (kaynak kopyaları)
    kaydedilen_modeller/     <- best_model_balanced_crop74.pth
    sonuclar/                <- metrikler.json, esik_ayari.json
    data/test/{normal,abnormal}/  <- örnek test hacimleri (isteğe bağlı)

Kullanım:
    DizMR.exe             -> arayüzü başlatır; sunucu hazır olunca tarayıcıyı açar.
                             Zaten çalışıyorsa yeni kopya açmaz, tarayıcıyı açar.
    DizMR.exe --selftest  -> model + tahmin + Grad-CAM öz-kontrolü; sonucu
                             iletişim kutusu ve günlük ile bildirir.

Günlükler: <exe klasörü>\dizmr_gunluk.txt ve %LOCALAPPDATA%\DizMR\dizmr_gunluk.txt
"""

import ctypes
import datetime
import faulthandler
import os
import socket
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path
from urllib.request import urlopen

PORT_VARSAYILAN = 8501
YEREL_LOG_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "DizMR"


def kok_klasor():
    """Exe (frozen) modunda exe'nin klasörü, geliştirmede bu dosyanın klasörü."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def gunluk(kok, mesaj):
    """Zaman damgalı günlük satırını iki konuma birden yazar.

    Dağıtım klasörü yazılamıyorsa (izin, salt-okunur ortam) yerel
    AppData kopyası tanı imkânını korur.
    """
    satir = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {mesaj}\n"
    for hedef in (kok / "dizmr_gunluk.txt", YEREL_LOG_DIR / "dizmr_gunluk.txt"):
        try:
            hedef.parent.mkdir(parents=True, exist_ok=True)
            with open(hedef, "a", encoding="utf-8") as f:
                f.write(satir)
        except OSError:
            pass


def hata_kutusu(mesaj, baslik="DizMR — Hata", simge=0x10, timeout_ms=0):
    """Konsol olmadığı için mesajı Windows iletişim kutusuyla gösterir.

    timeout_ms > 0 ise kutu o süre sonunda kendiliğinden kapanır (MessageBoxTimeoutW);
    böylece --selftest gibi otomatik çalıştırmalarda süreç asılı kalmaz. Kullanıcı
    yine de görüp 'Tamam'a basabilir. timeout_ms=0 => klasik bloklayan kutu
    (ölümcül başlatma hataları kullanıcı görene dek kalsın diye).
    """
    try:
        if timeout_ms and timeout_ms > 0:
            # MessageBoxTimeoutW: belgelenmemiş ama tüm modern Windows'ta mevcut
            ctypes.windll.user32.MessageBoxTimeoutW(None, mesaj, baslik, simge, 0, int(timeout_ms))
        else:
            ctypes.windll.user32.MessageBoxW(None, mesaj, baslik, simge)
    except Exception:
        try:
            ctypes.windll.user32.MessageBoxW(None, mesaj, baslik, simge)
        except Exception:
            pass


def akislari_bagla(kok):
    """Konsolsuz (windowed) modda None olan stdout/stderr'i dosyaya bağlar.

    Streamlit/torch günlükleri kaybolmaz; faulthandler yorumlayıcı
    çökmelerinde de iz bırakır.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    akis = None
    for hedef in (kok / "dizmr_cikti.txt", YEREL_LOG_DIR / "dizmr_cikti.txt"):
        try:
            hedef.parent.mkdir(parents=True, exist_ok=True)
            akis = open(hedef, "a", encoding="utf-8", buffering=1)
            break
        except OSError:
            continue
    if akis is None:
        return
    if sys.stdout is None:
        sys.stdout = akis
    if sys.stderr is None:
        sys.stderr = akis
    try:
        faulthandler.enable(file=akis)
    except Exception:
        pass


def sunucu_sagligi(port):
    """localhost:port üzerinde sağlıklı bir Streamlit sunucusu var mı?"""
    try:
        with urlopen(f"http://localhost:{port}/_stcore/health", timeout=2) as r:
            return r.read().decode().strip() == "ok"
    except Exception:
        return False


def bos_port(baslangic=PORT_VARSAYILAN, adet=20):
    """Verilen porttan başlayarak bağlanılabilir ilk boş portu döndürür."""
    for port in range(baslangic, baslangic + adet):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return baslangic


def hazir_olunca_tarayici_ac(kok, port, saniye=180):
    """Arka planda sunucunun sağlık ucunu bekler; hazır olunca tarayıcıyı açar."""
    url = f"http://localhost:{port}"

    def bekle():
        son = time.time() + saniye
        while time.time() < son:
            if sunucu_sagligi(port):
                gunluk(kok, f"Sunucu hazır, tarayıcı açılıyor: {url}")
                webbrowser.open(url)
                return
            time.sleep(0.5)
        gunluk(kok, f"Sunucu {saniye} sn içinde hazır olmadı; tarayıcı açılmadı.")

    threading.Thread(target=bekle, daemon=True).start()


def selftest(kok):
    """Arayüz açmadan modeli yükler ve (varsa) örnek bir hacmi sınıflandırır."""
    import torch
    import ortak

    rapor = [f"Kök klasör : {kok}",
             f"Model yolu : {ortak.MODEL_PATH} -> "
             f"{'VAR' if ortak.MODEL_PATH.exists() else 'YOK'}",
             f"Cihaz      : {ortak.get_device()}"]

    device = ortak.get_device()
    model = ortak.load_model(device=device)
    with torch.no_grad():
        y = model(torch.zeros(1, 3, ortak.INPUT_SIZE, ortak.INPUT_SIZE, device=device))
    rapor.append(f"Model yüklendi, ileri geçiş çıktısı: {list(y.shape)}")

    ornek = None
    for cls in ortak.CLASSES:
        cdir = ortak.TEST_DIR / cls
        if cdir.exists():
            adaylar = sorted(d for d in cdir.iterdir() if d.is_dir())
            if adaylar:
                ornek = (adaylar[0], cls)
                break
    if ornek is not None:
        yol, cls = ornek
        r = ortak.predict_volume(str(yol), model=model, device=device)
        rapor.append(f"Örnek tahmin (orta-3): {yol.name} (gerçek: {cls}) -> "
                     f"{r['pred_name']}  P(abnormal)=%{r['probs']['abnormal']*100:.1f}")
        from gradcam import gradcam_for_volume
        arka, cam, _ = gradcam_for_volume(str(yol), model=model, device=device)
        rapor.append(f"Grad-CAM (orta-3) üretildi: {arka.shape}")

        # Yeni: merkezi kayan pencere yolu (önerilen ayarla)
        ay, _ = ortak.onerilen_kayan_ayar()
        rk = ortak.predict_volume_sliding(
            str(yol), model=model, device=device,
            method=ay["method"], merkez_orani=ay["merkez_orani"], step=ay["step"])
        arka2, cam2, _ = gradcam_for_volume(str(yol), model=model, device=device,
                                            target_class=rk["pred_label"],
                                            slice_files=rk["best_window"])
        rapor.append(f"Kayan pencere ({rk['n_windows']} pencere, merkez "
                     f"%{int(ay['merkez_orani']*100)}, {rk['method']}) -> "
                     f"{rk['pred_name']}  P(abnormal)=%{rk['probs']['abnormal']*100:.1f}")
        rapor.append(f"Grad-CAM (en iyi pencere) üretildi: {arka2.shape}")
    else:
        rapor.append("Örnek hacim bulunamadı (data/test boş) — tahmin testi atlandı.")

    rapor.append("SELFTEST BAŞARILI")
    metin = "\n".join(rapor)
    print(metin)
    gunluk(kok, "SELFTEST:\n" + metin)
    hata_kutusu(metin, baslik="DizMR — Selftest", simge=0x40, timeout_ms=30000)


def arayuzu_baslat(kok, uygulama):
    # Tek kopya: uygulama zaten çalışıyorsa yeni sunucu açma, tarayıcıya götür.
    if sunucu_sagligi(PORT_VARSAYILAN):
        gunluk(kok, "Çalışan kopya bulundu; tarayıcı mevcut arayüze açılıyor.")
        webbrowser.open(f"http://localhost:{PORT_VARSAYILAN}")
        return

    port = bos_port()
    gunluk(kok, f"Arayüz başlatılıyor (port {port}).")
    hazir_olunca_tarayici_ac(kok, port)

    from streamlit.web import cli as stcli
    sys.argv = [
        "streamlit", "run", str(uygulama / "arayuz.py"),
        "--global.developmentMode=false",
        f"--server.port={port}",
        "--server.headless=true",          # tarayıcıyı biz açıyoruz
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
    ]
    stcli.main()


def main():
    kok = kok_klasor()
    os.environ.setdefault("DIZMR_KOK", str(kok))
    os.environ.setdefault("MPLBACKEND", "Agg")
    akislari_bagla(kok)

    uygulama = kok / "uygulama"
    sys.path.insert(0, str(uygulama))
    gunluk(kok, f"Başlatıldı (frozen={getattr(sys, 'frozen', False)}, "
                f"argv={sys.argv[1:]}, kok={kok})")

    try:
        if "--selftest" in sys.argv:
            selftest(kok)
        else:
            arayuzu_baslat(kok, uygulama)
        gunluk(kok, "Normal şekilde sonlandı.")
    except KeyboardInterrupt:
        gunluk(kok, "Ctrl+C ile kapatıldı.")
    except SystemExit:
        raise
    except Exception:
        hata = traceback.format_exc()
        gunluk(kok, "HATA:\n" + hata)
        ipucu = ""
        if "10106" in hata:
            ipucu = ("\n\nOlası çözüm: Yönetici PowerShell'de 'netsh winsock reset' "
                     "çalıştırıp bilgisayarı yeniden başlatın.")
        hata_kutusu("DizMR başlatılamadı.\n\n" + hata[-800:] + ipucu +
                    f"\n\nGünlük: {kok / 'dizmr_gunluk.txt'}")
        sys.exit(1)


if __name__ == "__main__":
    main()
