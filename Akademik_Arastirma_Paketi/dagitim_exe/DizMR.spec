# -*- mode: python ; coding: utf-8 -*-
# DizMR.spec — PyInstaller derleme tanımı (onedir)
#
# Derleme: derle.ps1 scripti ile yapılır (bkz. README.md).
# arayuz.py/ortak.py/gradcam.py PYZ arşivine GÖMÜLMEZ; dist içindeki
# uygulama/ klasöründen çalışma zamanında yüklenir (derle.ps1 kopyalar).
# Bu yüzden onların bağımlılıkları hiddenimports ile açıkça bildirilir.

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ("streamlit",):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

hiddenimports += [
    "torch",
    "torchvision",
    "torchvision.transforms",
    "torchvision.models",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.backends.backend_agg",
    "PIL",
    "PIL.Image",
    "numpy",
]

a = Analysis(
    ["launcher.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "pytest", "IPython", "jupyter"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DizMR",
    debug=False,
    strip=False,
    upx=False,
    console=False,  # konsolsuz: kullanıcı yalnızca tarayıcıdaki arayüzü görür;
                    # günlükler dizmr_gunluk.txt / dizmr_cikti.txt dosyalarına yazılır
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="DizMR",
)
