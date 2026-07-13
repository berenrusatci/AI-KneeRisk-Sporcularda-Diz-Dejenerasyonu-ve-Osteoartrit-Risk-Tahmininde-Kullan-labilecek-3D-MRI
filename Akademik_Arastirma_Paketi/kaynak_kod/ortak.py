# -*- coding: utf-8 -*-
"""
ortak.py — Merkezi Modül (Tek Kaynak)
=====================================

Bu modül, tüm değerlendirme / analiz / arayüz scriptlerinin ortak kullandığı
tek doğruluk kaynağıdır. Model mimarisi, veri seti sınıfı, ön işleme (transform)
zinciri ve dosya yolları YALNIZCA burada tanımlanır; diğer scriptler bunları
buradan içe aktarır (kopya-yapıştır YOK).

KRİTİK: Ön işleme zinciri, `best_model_balanced_crop74.pth` modelinin eğitimde
gördüğü işlem sırasıyla BİREBİR aynıdır. Değiştirmek doğruluğu düşürür.
Kaynak referans: convext-tiny-focalloss-centercrop-001(t).py

  2.5D giriş  : hacimdeki ortadaki 3 kesit (mid-1, mid, mid+1), her biri gri
                tonlama (L), 3 kanal olarak torch.stack -> [3, H, W]
  Transform   : CenterCrop(280) -> Resize(224,224, antialias) -> Normalize(ImageNet)
  Mimari      : torchvision convnext_tiny, classifier[2] = Dropout(0.5)+Linear(768,2)
  Sınıflar    : 0 = normal, 1 = abnormal
"""

import os
import re
import sys
import glob
from pathlib import Path

# Windows konsolu (cp1254) bazı Unicode karakterleri (∩, ✓, ✗) basamaz;
# çıktı UTF-8'e ayarlanır, basılamayan karakterler için 'replace' kullanılır.
# (Dosyalara yazım her zaman ayrıca UTF-8 açık yapılır.)
for _akis in (sys.stdout, sys.stderr):
    try:
        _akis.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, models
from PIL import Image

# ============================================================
# 1. YOLLAR (Proje yapısına göre otomatik çözümlenir)
# ============================================================
# kaynak_kod/ortak.py -> parents[1] = Akademik_Arastirma_Paketi
#                     -> parents[2] = Tubitak-Proje-2209A-Spor-ve-Saglik-Diz-MR (PROJE KÖKÜ)
# Dağıtım (exe) modunda DIZMR_KOK ortam değişkeni her iki kökü de tek klasöre
# sabitler; alt klasör adları (kaydedilen_modeller, sonuclar, data) aynı kalır.
_DIZMR_KOK = os.environ.get("DIZMR_KOK")
if _DIZMR_KOK:
    PAKET_KOK = Path(_DIZMR_KOK).resolve()
    PROJE_KOK = PAKET_KOK
else:
    PAKET_KOK = Path(__file__).resolve().parents[1]
    PROJE_KOK = Path(__file__).resolve().parents[2]

DATA_DIR    = PROJE_KOK / "data"
TRAIN_DIR   = DATA_DIR / "train"
VAL_DIR     = DATA_DIR / "val"
TEST_DIR    = DATA_DIR / "test"

MODEL_DIR   = PROJE_KOK / "kaydedilen_modeller"
MODEL_PATH  = MODEL_DIR / "best_model_balanced_crop74.pth"   # ana/dağıtılan model

SONUCLAR_DIR   = PAKET_KOK / "sonuclar"
GRADCAM_DIR    = SONUCLAR_DIR / "gradcam_ornekleri"
DOKUMAN_DIR    = PAKET_KOK / "dokumanlar"

# ============================================================
# 2. SABİTLER
# ============================================================
CLASSES        = ['normal', 'abnormal']                 # etiket sırası: 0, 1
DISPLAY_NAMES  = ['Normal (Sağlıklı)', 'Abnormal (Hasta)']
NUM_CLASSES    = 2
INPUT_SIZE     = 224
CENTER_CROP    = 280
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]

# Karar eşiği: P(abnormal) >= threshold => abnormal. 0.5 argmax'e eşdeğerdir.
DEFAULT_THRESHOLD = 0.5
ESIK_AYAR_PATH = SONUCLAR_DIR / "esik_ayari.json"

# Merkezi kayan pencere (tüm derinlik) önerilen ayarı — derinlik_tarama.py üretir.
# Uç kesitler dağılım-dışı olduğundan pencereleme hacmin ORTADAKİ merkez_orani'lık
# bölgesiyle sınırlanır; pencere skorları 'mean' ile birleştirilir. Bkz. derinlik_analizi.md.
KAYAN_AYAR_PATH = SONUCLAR_DIR / "derinlik_ayari.json"
KAYAN_VARSAYILAN = {"method": "mean", "merkez_orani": 0.3, "step": 2}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def onerilen_esik(varsayilan=DEFAULT_THRESHOLD):
    """esik_analizi.py tarafından üretilen önerilen (yüksek duyarlılık) eşiği okur.

    Dosya yoksa varsayılanı döndürür. Dönüş: (esik, bilgi_dict|None)
    """
    import json
    if ESIK_AYAR_PATH.exists():
        try:
            with open(ESIK_AYAR_PATH, encoding="utf-8") as f:
                d = json.load(f)
            return float(d.get("onerilen_esik", varsayilan)), d
        except Exception:
            pass
    return varsayilan, None


def onerilen_kayan_ayar():
    """derinlik_tarama.py tarafından seçilen önerilen kayan pencere ayarını okur.

    Dönüş: (ayar_dict {method, merkez_orani, step}, bilgi_dict|None). Dosya yoksa
    KAYAN_VARSAYILAN döner.
    """
    import json
    if KAYAN_AYAR_PATH.exists():
        try:
            with open(KAYAN_AYAR_PATH, encoding="utf-8") as f:
                d = json.load(f)
            ayar = {
                "method": d.get("method", KAYAN_VARSAYILAN["method"]),
                "merkez_orani": float(d.get("merkez_orani", KAYAN_VARSAYILAN["merkez_orani"])),
                "step": int(d.get("step", KAYAN_VARSAYILAN["step"])),
            }
            return ayar, d
        except Exception:
            pass
    return dict(KAYAN_VARSAYILAN), None


# ============================================================
# 3. ÖN İŞLEME (Eğitim/test ile birebir aynı — DEĞİŞTİRMEYİN)
# ============================================================
def build_transforms():
    """Değerlendirme/çıkarım için deterministik dönüşüm zinciri."""
    return transforms.Compose([
        transforms.CenterCrop(CENTER_CROP),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE), antialias=True),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# Modülü içe aktaran herkesin doğrudan kullanabilmesi için hazır örnek:
TEST_TRANSFORMS = build_transforms()


def natural_sort_key(s):
    """slice_2 < slice_10 olacak şekilde doğal (sayısal) sıralama anahtarı."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def list_volume_slices(sample_path):
    """Bir hacim klasöründeki tüm PNG kesitleri doğal sırayla döndürür.

    select_middle_slices'ın genel hâli: kayan pencere için önce tüm kesitler gerekir.
    """
    image_files = glob.glob(os.path.join(str(sample_path), "*.png"))
    image_files.sort(key=natural_sort_key)
    return image_files


def select_middle_slices(sample_path):
    """Bir hacim klasöründen ortadaki 3 kesitin dosya yollarını döndürür."""
    image_files = list_volume_slices(sample_path)
    if len(image_files) < 3:
        raise ValueError(f"En az 3 PNG kesit gerekli, bulunan: {len(image_files)} -> {sample_path}")
    mid = len(image_files) // 2
    return [image_files[mid - 1], image_files[mid], image_files[mid + 1]]


def window_baslangiclari(n_slices, window=3, step=1, merkez_orani=1.0):
    """Kayan pencere başlangıç indekslerini (mutlak, 0..n-window) döndürür.

    `merkez_orani` < 1.0 ise pencereleme hacmin yalnızca ortadaki bu orandaki
    kesitleriyle sınırlanır (uç/dağılım-dışı kesitler dışlanır). Örn. 0.5 -> orta
    %50. Son pencere sınırlı bölgenin sonuna hizalanır (kuyruk atlanmaz).

    select_windows ve tarama scripti bu TEK kaynağı kullanır (tutarlılık için).
    """
    if n_slices < window:
        raise ValueError(f"En az {window} kesit gerekli, bulunan: {n_slices}")
    if merkez_orani < 1.0:
        keep = min(n_slices, max(window, int(round(n_slices * merkez_orani))))
        ofset = (n_slices - keep) // 2
    else:
        keep, ofset = n_slices, 0

    son = ofset + keep - window                     # son geçerli başlangıç (mutlak)
    baslar = list(range(ofset, son + 1, step))
    if baslar[-1] != son:                            # kuyruğu kaçırmamak için ekle
        baslar.append(son)
    return baslar


def select_windows(sample_path, window=3, step=1, merkez_orani=1.0):
    """Hacmi ardışık `window`'lu kesit gruplarına (kayan pencere) böler.

    Model 3 kanal beklediği için window=3 sabittir; `step` granülerliği,
    `merkez_orani` ise yalnızca orta bölgenin kullanılıp kullanılmayacağını belirler
    (bkz. window_baslangiclari).

    Dönüş: her biri `window` adet dosya yolu içeren listelerin listesi.
    """
    files = list_volume_slices(sample_path)
    if len(files) < window:
        raise ValueError(f"En az {window} PNG kesit gerekli, bulunan: {len(files)} -> {sample_path}")
    baslar = window_baslangiclari(len(files), window=window, step=step, merkez_orani=merkez_orani)
    return [files[b:b + window] for b in baslar]


def stack_slices(slice_files):
    """3 gri tonlamalı kesiti [3, H, W] tensörü olarak yığar (transform ÖNCESİ)."""
    slices = []
    for img_path in slice_files:
        img = Image.open(img_path).convert('L')
        img_tensor = transforms.ToTensor()(img)   # [1, H, W], 0..1
        slices.append(img_tensor.squeeze(0))       # [H, W]
    return torch.stack(slices, dim=0)              # [3, H, W]


def load_volume_tensor(sample_path, transform=None):
    """Hacim klasörünü -> (dönüştürülmüş tensor [3,224,224], kullanılan 3 dosya).

    transform=None ise TEST_TRANSFORMS kullanılır.
    """
    slice_files = select_middle_slices(sample_path)
    img_25d = stack_slices(slice_files)
    tf = TEST_TRANSFORMS if transform is None else transform
    if tf is not None:
        img_25d = tf(img_25d)
    return img_25d, slice_files


def load_window_tensor(slice_files, transform=None):
    """Verilen 3 kesit yolundan -> (dönüştürülmüş tensor [3,224,224], kullanılan dosyalar).

    load_volume_tensor'ın belirli-pencere sürümü; kayan pencere modunda Grad-CAM'i
    en yüksek skorlu pencereye uygulamak için kullanılır.
    """
    img_25d = stack_slices(slice_files)
    tf = TEST_TRANSFORMS if transform is None else transform
    if tf is not None:
        img_25d = tf(img_25d)
    return img_25d, list(slice_files)


# ============================================================
# 4. VERİ SETİ (2.5D — sabit orta kesit, jitter kapalı)
# ============================================================
class Medical25DDataset(Dataset):
    """
    Klasör yapısı:  root_dir / {normal, abnormal} / <hacim_klasoru> / slice_XX.png
    Her örnek = bir hacim; ortadaki 3 kesit 3 kanal olarak kullanılır.

    return_path=True ise (tensor, label, sample_path) döndürür.
    """
    def __init__(self, root_dir, transform=None, return_path=False):
        self.root_dir = str(root_dir)
        self.transform = TEST_TRANSFORMS if transform is None else transform
        self.return_path = return_path
        self.classes = CLASSES
        self.samples = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for sample_folder in sorted(os.listdir(class_dir)):
                sample_path = os.path.join(class_dir, sample_folder)
                if os.path.isdir(sample_path):
                    png_files = glob.glob(os.path.join(sample_path, "*.png"))
                    if len(png_files) >= 3:
                        self.samples.append((sample_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, label = self.samples[idx]
        img_25d, _ = load_volume_tensor(sample_path, transform=self.transform)
        if self.return_path:
            return img_25d, label, sample_path
        return img_25d, label


# ============================================================
# 5. MODEL MİMARİSİ
# ============================================================
def build_model(num_classes=NUM_CLASSES):
    """ConvNeXt-Tiny + değiştirilmiş sınıflandırma başı (eğitimle birebir)."""
    model = models.convnext_tiny(weights=None)
    num_ftrs = model.classifier[2].in_features   # 768
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes),
    )
    return model


def load_model(model_path=None, device=None, num_classes=NUM_CLASSES):
    """Ağırlıkları yükler ve eval() modunda modeli döndürür."""
    device = get_device() if device is None else device
    model_path = MODEL_PATH if model_path is None else model_path
    model = build_model(num_classes=num_classes).to(device)
    state = torch.load(str(model_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# ============================================================
# 6. TEK HACİM ÇIKARIM (UI ve CLI ortak kullanır)
# ============================================================
@torch.no_grad()
def predict_volume(sample_path, model=None, device=None, transform=None, threshold=DEFAULT_THRESHOLD):
    """
    Bir hacim klasörü için tahmin döndürür.

    threshold : P(abnormal) >= threshold ise 'abnormal' (1), aksi halde 'normal' (0).
                Varsayılan 0.5 argmax ile aynıdır. Eşiği düşürmek DUYARLILIĞI artırır
                (daha çok hasta yakalanır), özgüllüğü düşürür.

    Dönüş (dict):
      pred_label   : 0/1  (eşiğe göre)
      pred_name    : 'Normal (Sağlıklı)' / 'Abnormal (Hasta)'
      probs        : {'normal': p0, 'abnormal': p1}  (softmax)
      confidence   : tahmin edilen sınıfın olasılığı (0..1)
      threshold    : kullanılan karar eşiği
      slice_files  : kullanılan 3 kesitin yolları
      logits       : ham çıktı (list)
    """
    device = get_device() if device is None else device
    if model is None:
        model = load_model(device=device)

    img_25d, slice_files = load_volume_tensor(sample_path, transform=transform)
    x = img_25d.unsqueeze(0).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0].cpu()
    p_abnormal = float(probs[1])
    pred_label = 1 if p_abnormal >= threshold else 0

    return {
        "pred_label": pred_label,
        "pred_name": DISPLAY_NAMES[pred_label],
        "probs": {CLASSES[0]: float(probs[0]), CLASSES[1]: p_abnormal},
        "confidence": float(probs[pred_label]),
        "threshold": float(threshold),
        "slice_files": slice_files,
        "logits": logits[0].detach().cpu().tolist(),
    }


# ============================================================
# 7. KAYAN PENCERE ÇIKARIM (tüm derinlik — çıkarım-zamanı birleştirme)
# ============================================================
# Model yalnızca orta 3 kesitle eğitildiği için, hacmin tamamını kullanmanın
# yeniden-eğitimsiz yolu: 3'erli kayan pencerelerle tüm hacmi tarayıp pencere-başı
# P(abnormal) skorlarını tek bir hacim skoruna birleştirmektir (MRNet tarzı).
AGGREGATION_METHODS = ("mean", "max", "topk")


def aggregate_scores(window_scores, method="mean", k=3):
    """Pencere-başı P(abnormal) dizisini tek hacim skoruna indirger.

      mean : tüm pencerelerin ortalaması (gürültüye dayanıklı, bulguyu sulandırabilir)
      max  : en yüksek pencere (=top-1; en duyarlı ama yanlış alarma açık)
      topk : en yüksek k pencerenin ortalaması (max ile mean arası denge)

    k, pencere sayısına kırpılır. Dönüş: float (0..1).
    """
    s = torch.as_tensor(window_scores, dtype=torch.float32).flatten()
    if s.numel() == 0:
        return 0.0
    if method == "mean":
        return float(s.mean())
    if method == "max":
        return float(s.max())
    if method == "topk":
        kk = max(1, min(int(k), s.numel()))
        return float(s.topk(kk).values.mean())
    raise ValueError(f"Bilinmeyen birleştirme yöntemi: {method!r} "
                     f"(geçerli: {AGGREGATION_METHODS})")


@torch.no_grad()
def sliding_window_scores(sample_path, model=None, device=None, transform=None,
                          window=3, step=1, merkez_orani=1.0, batch_size=64):
    """Bir hacim için pencere-başı P(abnormal) skorlarını hesaplar (birleştirmeden).

    Tüm pencereler tek (veya birkaç) batch hâlinde modele verilir. Dönüş:
    (window_scores: list[float], windows: list[list[str]]).
    """
    device = get_device() if device is None else device
    if model is None:
        model = load_model(device=device)
    tf = TEST_TRANSFORMS if transform is None else transform

    windows = select_windows(sample_path, window=window, step=step, merkez_orani=merkez_orani)
    tensors = []
    for w in windows:
        x = stack_slices(w)                 # [3, H, W] (transform öncesi)
        tensors.append(tf(x) if tf is not None else x)

    scores = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i:i + batch_size], dim=0).to(device)
        probs = F.softmax(model(batch), dim=1)[:, 1]   # P(abnormal) per pencere
        scores.extend(probs.cpu().tolist())
    return scores, windows


@torch.no_grad()
def predict_volume_sliding(sample_path, model=None, device=None, transform=None,
                           method="mean", k=3, threshold=DEFAULT_THRESHOLD,
                           window=3, step=1, merkez_orani=1.0, batch_size=64):
    """Kayan pencere + birleştirme ile tek hacim tahmini.

    predict_volume ile AYNI sözlük şemasını döndürür (arayüz aynı gösterim koduyla
    çalışsın diye), artı kayan-pencereye özgü alanlar:
      window_scores : pencere-başı P(abnormal) listesi
      method, k     : kullanılan birleştirme
      n_windows     : pencere sayısı
      best_window   : en yüksek skorlu pencerenin (Grad-CAM için) kesit yolları
    slice_files : en yüksek skorlu pencerenin 3 kesiti (arayüzdeki "modele giren
                  kesitler" gösterimiyle uyumlu).
    """
    device = get_device() if device is None else device
    if model is None:
        model = load_model(device=device)

    scores, windows = sliding_window_scores(
        sample_path, model=model, device=device, transform=transform,
        window=window, step=step, merkez_orani=merkez_orani, batch_size=batch_size)

    p_abnormal = aggregate_scores(scores, method=method, k=k)
    pred_label = 1 if p_abnormal >= threshold else 0
    en_iyi = int(max(range(len(scores)), key=lambda j: scores[j])) if scores else 0
    best_window = windows[en_iyi] if windows else []

    return {
        "pred_label": pred_label,
        "pred_name": DISPLAY_NAMES[pred_label],
        "probs": {CLASSES[0]: float(1.0 - p_abnormal), CLASSES[1]: float(p_abnormal)},
        "confidence": float(p_abnormal if pred_label == 1 else 1.0 - p_abnormal),
        "threshold": float(threshold),
        "slice_files": best_window,
        "window_scores": [float(s) for s in scores],
        "method": method,
        "k": int(k),
        "n_windows": len(scores),
        "best_window": best_window,
    }


if __name__ == "__main__":
    # Hızlı öz-kontrol
    print("PROJE_KOK :", PROJE_KOK)
    print("DATA_DIR  :", DATA_DIR, "->", "VAR" if DATA_DIR.exists() else "YOK")
    print("MODEL_PATH:", MODEL_PATH, "->", "VAR" if MODEL_PATH.exists() else "YOK")
    print("Cihaz     :", get_device())
