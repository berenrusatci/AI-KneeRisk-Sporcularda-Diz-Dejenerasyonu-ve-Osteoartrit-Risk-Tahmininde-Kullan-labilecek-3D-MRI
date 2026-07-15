# -*- coding: utf-8 -*-
"""
shap_analizi.py — SHAP Açıklanabilirlik (Explainability)
========================================================

Başvuruda Grad-CAM'in yanında taahhüt edilen ikinci XAI yöntemi. SHAP
(SHapley Additive exPlanations; Lundberg & Lee, 2017) her pikselin modelin
"abnormal" kararına katkısını oyun-teorik bir temelle niceler.

Görüntü/CNN girdisi için `shap.GradientExplainer` kullanılır: bir arka plan
(background) örneklem kümesine göre beklenen gradyanlar (expected gradients)
üzerinden SHAP değerleri hesaplanır. Değerler 3 kanal (2.5D komşu kesitler)
üzerinden toplanarak 224x224 bir katkı haritasına indirgenir ve orta kesit
üzerine bindirilir (Grad-CAM ile karşılaştırılabilir gösterim).

  Kırmızı  (+) : abnormal olasılığını ARTIRAN bölgeler
  Mavi     (-) : abnormal olasılığını AZALTAN bölgeler

İki kullanım:
  1) Doğrudan çalıştırma -> her sınıftan birkaç örnek için overlay üretir:
       sonuclar/shap_ornekleri/*.png
  2) İçe aktarma (arayuz.py) -> shap_for_volume() ile tek hacim için
     (arka plan, imzalı katkı haritası, tahmin bilgisi) döndürür.

Kullanım:
    python shap_analizi.py [--ornek-sayisi 3] [--arka-plan 24]
"""

import os
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap

import ortak


def _denorm_orta_kanal(x_tensor):
    """Dönüştürülmüş [3,224,224] girdinin orta kanalını görüntü için geri normalize eder."""
    c = 1
    img = x_tensor[c].cpu().numpy() * ortak.IMAGENET_STD[c] + ortak.IMAGENET_MEAN[c]
    return np.clip(img, 0, 1)


def _ornek_yollari(n_arka_plan):
    """Arka plan için test setinden dengeli (normal+abnormal) hacim yolları toplar."""
    yollar = []
    for cls in ortak.CLASSES:
        cdir = ortak.TEST_DIR / cls
        if cdir.exists():
            folders = sorted(os.listdir(cdir))
            for f in folders[: max(1, n_arka_plan // 2)]:
                p = cdir / f
                if p.is_dir():
                    yollar.append(p)
    return yollar


def arka_plan_tensor(n_arka_plan, device):
    """Arka plan (referans) örneklem tensörünü [K,3,224,224] oluşturur."""
    tensors = []
    for p in _ornek_yollari(n_arka_plan):
        x, _ = ortak.load_volume_tensor(p)
        tensors.append(x)
    if not tensors:
        raise RuntimeError("Arka plan için test hacmi bulunamadı.")
    return torch.stack(tensors, dim=0).to(device)


def _shap_haritasi(shap_values, class_idx=1):
    """GradientExplainer çıktısını tek örnek için imzalı 224x224 haritaya indirger.

    shap sürümüne göre çıktı: list[array(N,3,H,W)] (sınıf başına) VEYA
    array(N,3,H,W,n_classes). Her iki biçim de desteklenir. Kanallar toplanır.
    """
    if isinstance(shap_values, list):
        sv = shap_values[class_idx]              # (N,3,H,W)
        arr = np.asarray(sv)[0]                   # (3,H,W)
    else:
        arr = np.asarray(shap_values)[0]          # (3,H,W,n_classes) veya (3,H,W)
        if arr.ndim == 4:
            arr = arr[..., class_idx]             # (3,H,W)
    return arr.sum(axis=0)                         # (H,W) — kanallar (kesitler) toplandı


def shap_for_volume(sample_path, explainer=None, model=None, device=None,
                    slice_files=None, class_idx=1, n_arka_plan=24):
    """
    Tek hacim için SHAP katkı haritası.

    explainer verilmezse geçici bir GradientExplainer kurulur (tek örnek için
    yavaştır; toplu üretimde main() paylaşımlı explainer kullanır).

    Dönüş:
      arka_plan (2D 0..1), shap_harita (2D imzalı), bilgi (dict: pred_label, probs, ...)
    """
    device = ortak.get_device() if device is None else device
    if model is None:
        model = ortak.load_model(device=device)
    model.eval()

    if slice_files:
        x_t, slice_files = ortak.load_window_tensor(slice_files)
    else:
        x_t, slice_files = ortak.load_volume_tensor(sample_path)
    x = x_t.unsqueeze(0).to(device)

    if explainer is None:
        bg = arka_plan_tensor(n_arka_plan, device)
        explainer = shap.GradientExplainer(model, bg)

    shap_values = explainer.shap_values(x)
    harita = _shap_haritasi(shap_values, class_idx=class_idx)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu()
    pred_label = int(torch.argmax(probs).item())

    bilgi = {
        "pred_label": pred_label,
        "pred_name": ortak.DISPLAY_NAMES[pred_label],
        "shap_class": class_idx,
        "shap_class_name": ortak.DISPLAY_NAMES[class_idx],
        "probs": {ortak.CLASSES[0]: float(probs[0]), ortak.CLASSES[1]: float(probs[1])},
        "slice_files": slice_files,
    }
    return _denorm_orta_kanal(x_t), harita, bilgi


def overlay_kaydet(arka_plan, harita, out_path, baslik=""):
    """Orta kesit üzerine imzalı SHAP katkısını (diverging) bindirir."""
    lim = float(np.abs(harita).max()) + 1e-8
    plt.figure(figsize=(6, 6), facecolor="white")
    plt.imshow(arka_plan, cmap="gray")
    plt.imshow(harita, cmap="bwr", alpha=0.55, vmin=-lim, vmax=lim)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label("SHAP katkısı (kırmızı: +abnormal)", fontsize=8)
    plt.title(baslik, fontsize=11)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="SHAP örnek görselleri üret")
    parser.add_argument("--ornek-sayisi", type=int, default=3,
                        help="Her sınıftan üretilecek örnek sayısı")
    parser.add_argument("--arka-plan", type=int, default=24,
                        help="SHAP arka plan (referans) örneklem büyüklüğü")
    args = parser.parse_args()

    out_dir = ortak.SONUCLAR_DIR / "shap_ornekleri"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = ortak.get_device()
    print(f"[*] Cihaz: {device}")
    model = ortak.load_model(device=device)

    print(f"[*] SHAP arka planı hazırlanıyor (K={args.arka_plan})...")
    bg = arka_plan_tensor(args.arka_plan, device)
    explainer = shap.GradientExplainer(model, bg)

    for cls_idx, cls in enumerate(ortak.CLASSES):
        cdir = ortak.TEST_DIR / cls
        if not cdir.exists():
            continue
        # arka planla örtüşmeyi azaltmak için sondan örnek seç
        ornekler = sorted(os.listdir(cdir))[-args.ornek_sayisi:]
        for i, folder in enumerate(ornekler):
            spath = cdir / folder
            arka, harita, bilgi = shap_for_volume(
                spath, explainer=explainer, model=model, device=device, class_idx=1)
            dogru = "DOGRU" if bilgi["pred_label"] == cls_idx else "YANLIS"
            baslik = (f"Gerçek: {ortak.DISPLAY_NAMES[cls_idx]}\n"
                      f"Tahmin: {bilgi['pred_name']} "
                      f"(%{bilgi['probs'][ortak.CLASSES[bilgi['pred_label']]]*100:.1f}) [{dogru}]")
            out = out_dir / f"shap_{cls}_{i+1}.png"
            overlay_kaydet(arka, harita, out, baslik)
            print(f"[+] {out.name}  <- {folder}  ({dogru})")

    print(f"\n[+] SHAP örnekleri: {out_dir}")


if __name__ == "__main__":
    main()
