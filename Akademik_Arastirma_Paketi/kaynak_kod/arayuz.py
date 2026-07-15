# -*- coding: utf-8 -*-
"""
arayuz.py — Diz MR Sınıflandırma Demo Arayüzü (Streamlit)
=========================================================

Çalıştırma:
    & "C:\\Projelerim\\Tubitak-Proje\\.venv312\\Scripts\\streamlit.exe" run arayuz.py

Özellikler:
  - Test setinden bir hasta hacmi seçme VEYA kendi PNG kesitlerini yükleme
  - Model tahmini + sınıf olasılıkları (güven çubuğu)
  - Modele giren 3 kesitin gösterimi
  - Grad-CAM ısı haritası (modelin odaklandığı bölge)
  - Yan panelde model kartı özeti ve test seti metrikleri
"""

import os
import json
import tempfile
import threading

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

import ortak
from gradcam import gradcam_for_volume

st.set_page_config(page_title="Diz MR Sınıflandırma", page_icon="🦵", layout="wide")


# ------------------------------------------------------------
# Önbelleğe alınan kaynaklar
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Model yükleniyor...")
def get_model():
    device = ortak.get_device()
    model = ortak.load_model(device=device)
    return model, device


@st.cache_data
def get_metrikler():
    p = ortak.SONUCLAR_DIR / "metrikler.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def test_hacim_listesi(cls):
    cdir = ortak.TEST_DIR / cls
    if cdir.exists():
        return sorted([d for d in os.listdir(cdir) if os.path.isdir(cdir / d)])
    return []


# ------------------------------------------------------------
# Yan panel — Model kartı & metrikler
# ------------------------------------------------------------
def yan_panel():
    st.sidebar.header("📋 Model Bilgisi")
    st.sidebar.markdown(
        "- **Mimari:** ConvNeXt-Tiny (2.5D)\n"
        "- **Görev:** Diz MR — Normal / Abnormal\n"
        "- **Giriş:** Orta 3 kesit, 224×224\n"
        "- **Kayıp:** Focal Loss + sınıf dengesi\n"
    )
    m = get_metrikler()
    st.sidebar.header("📊 Test Seti Metrikleri")
    if m:
        st.sidebar.metric("Doğruluk (Accuracy)", f"%{m['accuracy']*100:.1f}")
        c1, c2 = st.sidebar.columns(2)
        c1.metric("Duyarlılık", f"%{m['duyarlilik_sensitivity']*100:.1f}")
        c2.metric("Özgüllük", f"%{m['ozgulluk_specificity']*100:.1f}")
        c3, c4 = st.sidebar.columns(2)
        c3.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
        c4.metric("PR-AUC", f"{m['pr_auc_average_precision']:.3f}")
        st.sidebar.caption(f"Test örnek sayısı: {m['test_ornek_sayisi']}")
    else:
        st.sidebar.info("metrikler.json bulunamadı. Önce degerlendirme.py çalıştırın.")
    st.sidebar.divider()
    st.sidebar.caption("⚠️ Bu araç akademik/araştırma amaçlıdır, klinik tanı aracı değildir.")

    # Exe (DizMR.exe) dağıtımında konsol penceresi olmadığından uygulama
    # buradan kapatılır; normal (streamlit run) kullanımda görünmez.
    if os.environ.get("DIZMR_KOK"):
        st.sidebar.divider()
        if st.sidebar.button("🛑 Uygulamayı Kapat"):
            st.sidebar.success("Uygulama kapatıldı. Bu sekmeyi kapatabilirsiniz.")
            threading.Timer(0.5, lambda: os._exit(0)).start()


# ------------------------------------------------------------
# Tahmin sonucu gösterimi
# ------------------------------------------------------------
def esik_secici():
    """Yan panelde karar eşiği modu seçtirir; (eşik, mod_etiketi) döndürür."""
    oneri, bilgi = ortak.onerilen_esik()
    st.sidebar.header("🎚️ Karar Eşiği")
    secenekler = ["Standart (0.50)",
                  f"Yüksek Duyarlılık ({oneri:.2f}) — önerilen",
                  "Özel"]
    sec = st.sidebar.radio("Mod", secenekler, index=0, label_visibility="collapsed")
    if sec.startswith("Standart"):
        t, mod = 0.50, "Standart"
    elif sec.startswith("Yüksek"):
        t, mod = oneri, "Yüksek Duyarlılık"
        if bilgi:
            tm = bilgi["test_metrikleri"]
            st.sidebar.caption(
                f"Test: Duyarlılık **%{tm['duyarlilik']*100:.0f}**, "
                f"Özgüllük **%{tm['ozgulluk']*100:.0f}** (kaçan hasta: {tm['FN']})")
    else:
        t, mod = st.sidebar.slider("Eşik P(abnormal)", 0.05, 0.95, 0.50, 0.01), "Özel"
    st.sidebar.caption("↓ Eşik = daha çok hasta yakalanır (duyarlılık↑), "
                       "yanlış alarm artar (özgüllük↓).")
    return t, mod


def kesit_modu_secici():
    """Yan panelde kesit modunu (orta-3 vs merkezi kayan pencere) seçtirir.

    Dönüş: (mod_etiketi, kayan_ayar_dict). Kayan pencere ayarı derinlik_tarama.py
    tarafından üretilen sonuclar/derinlik_ayari.json'dan okunur (yoksa varsayılan).
    """
    ayar, bilgi = ortak.onerilen_kayan_ayar()
    st.sidebar.header("🧩 Kesit Modu")
    secenekler = ["Orta 3 kesit (temel)", "Merkezi kayan pencere (önerilen)"]
    sec = st.sidebar.radio("Kesit modu", secenekler, index=0, label_visibility="collapsed")
    if sec.startswith("Merkezi"):
        st.sidebar.caption(
            f"Hacmin ortadaki **%{int(ayar['merkez_orani']*100)}** bölgesi 3'erli "
            f"pencerelerle taranıp skorlar **ortalanır** (uç kesitler dışlanır).")
        if bilgi and "test_metrikleri" in bilgi:
            tm = bilgi["test_metrikleri"]
            st.sidebar.caption(
                f"Test: ROC-AUC **{bilgi.get('test_roc', 0):.3f}**, "
                f"Doğruluk **%{tm.get('accuracy', 0)*100:.1f}** "
                f"(temel: {bilgi.get('temel_test_roc', 0):.3f} / "
                f"%{bilgi.get('temel_test_acc', 0)*100:.1f})")
    else:
        st.sidebar.caption("Yalnızca hacmin ortadaki 3 kesiti kullanılır (eğitimdeki giriş).")
    return sec, ayar


def sonucu_goster(hacim_yolu, model, device, gercek_etiket=None, threshold=0.50, mod="Standart",
                  kesit_modu="Orta 3 kesit (temel)", kayan_ayar=None):
    kayan = kesit_modu.startswith("Merkezi")
    with st.spinner("Analiz ediliyor..."):
        if kayan:
            ay = kayan_ayar or ortak.KAYAN_VARSAYILAN
            r = ortak.predict_volume_sliding(
                hacim_yolu, model=model, device=device, threshold=threshold,
                method=ay["method"], merkez_orani=ay["merkez_orani"], step=ay["step"])
            arka, cam, _ = gradcam_for_volume(hacim_yolu, model=model, device=device,
                                              target_class=r["pred_label"],
                                              slice_files=r["best_window"])
        else:
            r = ortak.predict_volume(hacim_yolu, model=model, device=device, threshold=threshold)
            arka, cam, _ = gradcam_for_volume(hacim_yolu, model=model, device=device,
                                              target_class=r["pred_label"])

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔍 Tahmin Sonucu")
        renk = "🟢" if r["pred_label"] == 0 else "🔴"
        st.markdown(f"### {renk} {r['pred_name']}")
        st.progress(float(r["probs"]["abnormal"]),
                    text=f"P(abnormal): %{r['probs']['abnormal']*100:.1f}  ·  "
                         f"eşik: {threshold:.2f} ({mod})")
        if kayan:
            st.caption(f"🧩 Merkezi kayan pencere · {r['n_windows']} pencere "
                       f"'{r['method']}' ile birleştirildi")

        st.write("**Sınıf Olasılıkları**")
        st.write(f"- Normal (Sağlıklı): **%{r['probs']['normal']*100:.1f}**")
        st.write(f"- Abnormal (Hasta): **%{r['probs']['abnormal']*100:.1f}**")

        # Risk skoru (0-100) ve grup (Düşük/Orta/Yüksek)
        grup_emoji = {"Düşük": "🟢", "Orta": "🟡", "Yüksek": "🔴"}
        st.metric("🧮 Dejenerasyon/OA Risk Skoru",
                  f"{r['risk_skoru']}/100",
                  help="Risk skoru = P(abnormal)×100. Gruplar: Düşük 0-33, Orta 34-66, Yüksek 67-100.")
        st.markdown(f"**Risk grubu:** {grup_emoji.get(r['risk_grubu'], '')} "
                    f"**{r['risk_grubu']}**")
        if mod != "Standart" and 0.5 > r["probs"]["abnormal"] >= threshold:
            st.info("ℹ️ Bu hacim standart eşikte (0.50) 'normal' sayılırdı; "
                    "düşük eşik sayesinde 'abnormal' olarak işaretlendi.")

        if gercek_etiket is not None:
            dogru = (r["pred_label"] == gercek_etiket)
            if dogru:
                st.success(f"✓ Doğru tahmin (Gerçek: {ortak.DISPLAY_NAMES[gercek_etiket]})")
            else:
                st.error(f"✗ Yanlış tahmin (Gerçek: {ortak.DISPLAY_NAMES[gercek_etiket]})")

        st.write("**En yüksek skorlu pencere (3 kesit)**" if kayan
                 else "**Modele giren 3 kesit**")
        kesit_kolonlari = st.columns(3)
        for i, sf in enumerate(r["slice_files"]):
            with kesit_kolonlari[i]:
                st.image(Image.open(sf).convert("L"),
                         caption=os.path.basename(sf), width="stretch")

    with col2:
        st.subheader("🌡️ Grad-CAM (Modelin Odağı)")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(arka, cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.axis("off")
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.caption("Kırmızı bölgeler, modelin kararında en etkili alanlardır. "
                   "Kenardaki siyah çerçeve CenterCrop(280) ön-işlemesinden gelir.")


# ------------------------------------------------------------
# Ana
# ------------------------------------------------------------
def main():
    st.title("🦵 Diz MR Sınıflandırma — Akademik Demo")
    st.caption("ConvNeXt-Tiny 2.5D · TÜBİTAK 2209-A · Normal / Abnormal diz MR ayrımı")

    if not ortak.MODEL_PATH.exists():
        st.error(f"Model dosyası bulunamadı: {ortak.MODEL_PATH}")
        st.stop()

    yan_panel()
    esik, mod = esik_secici()
    kesit_modu, kayan_ayar = kesit_modu_secici()
    model, device = get_model()

    sekme1, sekme2 = st.tabs(["🗂️ Test setinden seç", "⬆️ Kendi kesitlerini yükle"])

    # --- Sekme 1: Test setinden hacim seç ---
    with sekme1:
        c1, c2 = st.columns([1, 2])
        with c1:
            cls = st.selectbox("Sınıf", ortak.CLASSES,
                               format_func=lambda c: ortak.DISPLAY_NAMES[ortak.CLASSES.index(c)])
        with c2:
            hacimler = test_hacim_listesi(cls)
            if not hacimler:
                st.warning("Bu sınıf için test hacmi bulunamadı.")
            else:
                secim = st.selectbox(f"Hacim ({len(hacimler)} adet)", hacimler)
        if hacimler and st.button("Tahmin Et", type="primary", key="btn_test"):
            yol = ortak.TEST_DIR / cls / secim
            gercek = ortak.CLASSES.index(cls)
            sonucu_goster(str(yol), model, device, gercek_etiket=gercek,
                          threshold=esik, mod=mod,
                          kesit_modu=kesit_modu, kayan_ayar=kayan_ayar)

    # --- Sekme 2: Kendi kesitlerini yükle ---
    with sekme2:
        if kesit_modu.startswith("Merkezi"):
            st.write("En az **3 PNG kesit** yükleyin. **Merkezi kayan pencere** modunda "
                     "yüklenen hacmin ortadaki bölgesi 3'erli pencerelerle taranır.")
        else:
            st.write("En az **3 PNG kesit** yükleyin (ardışık kesitler önerilir; "
                     "model ortadaki 3 kesiti kullanır).")
        yuklenen = st.file_uploader("PNG kesitler", type=["png"],
                                    accept_multiple_files=True, key="uploader")
        if yuklenen and len(yuklenen) >= 3:
            if st.button("Tahmin Et", type="primary", key="btn_upload"):
                tmp = tempfile.mkdtemp(prefix="dizmr_")
                for i, uf in enumerate(sorted(yuklenen, key=lambda f: f.name)):
                    with open(os.path.join(tmp, f"slice_{i:03d}.png"), "wb") as out:
                        out.write(uf.getbuffer())
                sonucu_goster(tmp, model, device, gercek_etiket=None,
                              threshold=esik, mod=mod,
                              kesit_modu=kesit_modu, kayan_ayar=kayan_ayar)
        elif yuklenen:
            st.warning("En az 3 kesit gerekli.")


if __name__ == "__main__":
    main()
