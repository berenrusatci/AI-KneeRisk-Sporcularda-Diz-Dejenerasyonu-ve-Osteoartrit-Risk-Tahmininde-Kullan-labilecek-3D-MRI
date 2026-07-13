# -*- coding: utf-8 -*-
"""
gradcam.py — Grad-CAM Açıklanabilirlik (Explainability)
=======================================================

ConvNeXt-Tiny modelinin kararını verirken görüntünün HANGİ bölgelerine
odaklandığını gösteren ısı haritaları (Grad-CAM) üretir. Ekstra kütüphane
KULLANMADAN, modelin son özellik bloğuna forward/backward hook takılarak
uygulanır.

Hedef katman: model.features[-1]  (ConvNeXt son evre, çıktı [B, 768, 7, 7]).

İki kullanım:
  1) Doğrudan çalıştırma -> her sınıftan birkaç örnek için overlay üretir:
       sonuclar/gradcam_ornekleri/*.png
  2) İçe aktarma (arayuz.py) -> gradcam_for_volume() ile tek hacim için
     (arka plan, ısı haritası, tahmin bilgisi) döndürür.

Kullanım:
    python gradcam.py [--ornek-sayisi 3]
"""

import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ortak


def gradcam_hedef_katman(model):
    """ConvNeXt-Tiny için Grad-CAM hedef katmanı (son özellik bloğu)."""
    return model.features[-1]


class GradCAM:
    """Minimal Grad-CAM uygulaması (tek örnek için)."""

    def __init__(self, model, target_layer):
        self.model = model
        self.acts = None
        self.grads = None
        self._h1 = target_layer.register_forward_hook(self._fwd)
        self._h2 = target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.acts = out.detach()

    def _bwd(self, module, grad_in, grad_out):
        self.grads = grad_out[0].detach()

    def remove(self):
        self._h1.remove()
        self._h2.remove()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(True):
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())
            score = logits[0, class_idx]
            score.backward()

        # Grad-CAM: kanal ağırlıkları = gradyanların uzamsal ortalaması
        w = self.grads.mean(dim=(2, 3), keepdim=True)          # [1, C, 1, 1]
        cam = (w * self.acts).sum(dim=1, keepdim=True)          # [1, 1, h, w]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(ortak.INPUT_SIZE, ortak.INPUT_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), class_idx, logits.detach()


def _denorm_orta_kanal(x_tensor):
    """Dönüştürülmüş [3,224,224] girdinin orta kanalını görüntü için geri normalize eder."""
    c = 1  # orta kesit
    img = x_tensor[c].cpu().numpy() * ortak.IMAGENET_STD[c] + ortak.IMAGENET_MEAN[c]
    return np.clip(img, 0, 1)


def gradcam_for_volume(sample_path, model=None, device=None, target_class=None,
                       slice_files=None):
    """
    Tek hacim için Grad-CAM.

    slice_files verilirse (kayan pencere modu) Grad-CAM o belirli 3 kesit (en yüksek
    skorlu pencere) üzerinde hesaplanır; aksi halde hacmin ortadaki 3 kesiti kullanılır.

    Dönüş:
      arka_plan (2D 0..1), isı_haritası (2D 0..1), bilgi (dict: pred_label, probs, ...)
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

    cam_engine = GradCAM(model, gradcam_hedef_katman(model))
    try:
        cam, used_class, logits = cam_engine(x, class_idx=target_class)
    finally:
        cam_engine.remove()

    probs = F.softmax(logits, dim=1)[0].cpu()
    pred_label = int(torch.argmax(probs).item())
    arka_plan = _denorm_orta_kanal(x_t)

    bilgi = {
        "pred_label": pred_label,
        "pred_name": ortak.DISPLAY_NAMES[pred_label],
        "gradcam_class": used_class,
        "gradcam_class_name": ortak.DISPLAY_NAMES[used_class],
        "probs": {ortak.CLASSES[0]: float(probs[0]), ortak.CLASSES[1]: float(probs[1])},
        "slice_files": slice_files,
    }
    return arka_plan, cam, bilgi


def overlay_kaydet(arka_plan, cam, out_path, baslik=""):
    plt.figure(figsize=(6, 6), facecolor="white")
    plt.imshow(arka_plan, cmap="gray")
    plt.imshow(cam, cmap="jet", alpha=0.45)
    plt.title(baslik, fontsize=11)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM örnek görselleri üret")
    parser.add_argument("--ornek-sayisi", type=int, default=3,
                        help="Her sınıftan üretilecek örnek sayısı")
    args = parser.parse_args()

    ortak.GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    device = ortak.get_device()
    print(f"[*] Cihaz: {device}")
    model = ortak.load_model(device=device)

    for cls_idx, cls in enumerate(ortak.CLASSES):
        cdir = ortak.TEST_DIR / cls
        if not cdir.exists():
            continue
        ornekler = sorted(os.listdir(cdir))[:args.ornek_sayisi]
        for i, folder in enumerate(ornekler):
            spath = cdir / folder
            arka, cam, bilgi = gradcam_for_volume(spath, model=model, device=device)
            dogru = "DOGRU" if bilgi["pred_label"] == cls_idx else "YANLIS"
            baslik = (f"Gerçek: {ortak.DISPLAY_NAMES[cls_idx]}\n"
                      f"Tahmin: {bilgi['pred_name']} "
                      f"(%{bilgi['probs'][ortak.CLASSES[bilgi['pred_label']]]*100:.1f}) [{dogru}]")
            out = ortak.GRADCAM_DIR / f"gradcam_{cls}_{i+1}.png"
            overlay_kaydet(arka, cam, out, baslik)
            print(f"[+] {out.name}  <- {folder}  ({dogru})")

    print(f"\n[+] Grad-CAM örnekleri: {ortak.GRADCAM_DIR}")


if __name__ == "__main__":
    main()
