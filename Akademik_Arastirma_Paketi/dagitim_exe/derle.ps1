# ============================================================
# derle.ps1 — DizMR.exe derleme scripti
# ============================================================
# Kullanım (PowerShell):
#   cd .\Akademik_Arastirma_Paketi\dagitim_exe
#   powershell -ExecutionPolicy Bypass -File .\derle.ps1
#
# Gereksinim: C:\Projelerim\Tubitak-Proje\.venv_exe
#   (CPU-only torch + streamlit + matplotlib + pyinstaller kurulu ortam;
#    kurulum komutları README.md içindedir)
#
# Çıktı: .\dist\DizMR\  (DizMR.exe + tüm bağımlılıklar + model + örnek veri)

$ErrorActionPreference = "Stop"

$Burasi    = Split-Path -Parent $MyInvocation.MyCommand.Path
$PaketKok  = Split-Path -Parent $Burasi                       # Akademik_Arastirma_Paketi
$ProjeKok  = Split-Path -Parent $PaketKok                     # Tubitak-Proje-2209A-...
$KaynakKod = Join-Path $PaketKok "kaynak_kod"
$VenvPy    = "C:\Projelerim\Tubitak-Proje\.venv_exe\Scripts\python.exe"
$OrnekHacimSayisi = 3    # dist'e her sınıftan kopyalanacak örnek hacim sayısı

if (-not (Test-Path $VenvPy)) {
    Write-Error "Derleme ortamı bulunamadı: $VenvPy (kurulum için README.md'ye bakın)"
}

# --- 1. PyInstaller derlemesi ---
Write-Host "[1/3] PyInstaller derleniyor (birkaç dakika sürebilir)..." -ForegroundColor Cyan
Push-Location $Burasi
& $VenvPy -m PyInstaller DizMR.spec --noconfirm --clean
if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Error "PyInstaller derlemesi başarısız (çıkış kodu $LASTEXITCODE)" }
Pop-Location

$Dist = Join-Path $Burasi "dist\DizMR"

# --- 2. Uygulama dosyaları, model ve sonuç JSON'ları ---
Write-Host "[2/3] Model, kaynak kod ve sonuç dosyaları kopyalanıyor..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force (Join-Path $Dist "uygulama")            | Out-Null
New-Item -ItemType Directory -Force (Join-Path $Dist "kaydedilen_modeller") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $Dist "sonuclar")            | Out-Null

foreach ($f in "arayuz.py", "ortak.py", "gradcam.py") {
    Copy-Item (Join-Path $KaynakKod $f) (Join-Path $Dist "uygulama") -Force
}
Copy-Item (Join-Path $ProjeKok "kaydedilen_modeller\best_model_balanced_crop74.pth") `
          (Join-Path $Dist "kaydedilen_modeller") -Force
foreach ($f in "metrikler.json", "esik_ayari.json", "derinlik_ayari.json") {
    $p = Join-Path $PaketKok "sonuclar\$f"
    if (Test-Path $p) { Copy-Item $p (Join-Path $Dist "sonuclar") -Force }
}
Copy-Item (Join-Path $Burasi "KULLANIM.md") $Dist -Force

# --- 3. Örnek test hacimleri (demo için) ---
Write-Host "[3/3] Örnek test hacimleri kopyalanıyor..." -ForegroundColor Cyan
foreach ($cls in "normal", "abnormal") {
    $kaynak = Join-Path $ProjeKok "data\test\$cls"
    $hedef  = Join-Path $Dist "data\test\$cls"
    if (Test-Path $kaynak) {
        New-Item -ItemType Directory -Force $hedef | Out-Null
        Get-ChildItem $kaynak -Directory | Select-Object -First $OrnekHacimSayisi |
            ForEach-Object { Copy-Item $_.FullName $hedef -Recurse -Force }
    }
}

$Boyut = [math]::Round(((Get-ChildItem $Dist -Recurse -File | Measure-Object Length -Sum).Sum / 1MB))
Write-Host ""
Write-Host "[+] Derleme tamamlandı: $Dist ($Boyut MB)" -ForegroundColor Green
Write-Host "    Test  : .\dist\DizMR\DizMR.exe --selftest"
Write-Host "    Çalıştır: .\dist\DizMR\DizMR.exe"
