import os
import glob
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ==========================================
# 1. VERİ SETİ SINIFI (Test İçin Sabit Kesit)
# ==========================================
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

class Medical25DDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['normal', 'abnormal']
        self.samples = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for sample_folder in os.listdir(class_dir):
                sample_path = os.path.join(class_dir, sample_folder)
                if os.path.isdir(sample_path):
                    png_files = glob.glob(os.path.join(sample_path, "*.png"))
                    if len(png_files) >= 3:
                        self.samples.append((sample_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, label = self.samples[idx]
        image_files = glob.glob(os.path.join(sample_path, "*.png"))
        image_files.sort(key=natural_sort_key)
        
        # Test aşamasında her zaman tam ortadaki 3 kesiti alıyoruz (Jittering kapalı)
        mid_idx = len(image_files) // 2
        selected_files = [image_files[mid_idx - 1], image_files[mid_idx], image_files[mid_idx + 1]]

        slices = []
        for img_path in selected_files:
            img = Image.open(img_path).convert('L') 
            img_tensor = transforms.ToTensor()(img) 
            slices.append(img_tensor.squeeze(0))    

        img_25d = torch.stack(slices, dim=0)

        if self.transform:
            img_25d = self.transform(img_25d)
        return img_25d, label

# ==========================================
# 2. MODEL MİMARİSİ
# ==========================================
def build_model(num_classes=2):
    model = models.convnext_tiny(weights=None) 
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.5),  
        nn.Linear(num_ftrs, num_classes)
    )
    return model

# ==========================================
# 3. TEST VE RAPORLAMA ANA BLOKU
# ==========================================
if __name__ == '__main__':
    # Dosya Yolları
    test_dir  = r"C:\Projeler\tbtk2209a\proje_verisi_yeni\test"
    # YENİ MODEL DOSYASI BURAYA EKLENDİ
    model_path = r"C:\Projeler\tbtk2209a\kaydedilen_modeller\best_model_balanced_crop.pth"
    rapor_klasoru = r"C:\Projeler\tbtk2209a"

    BATCH_SIZE = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Test Donanımı: {device}")

    # EĞİTİMDEKİ ÖN İŞLEME BURAYA DA EKLENDİ (CenterCrop)
    test_transforms = transforms.Compose([
        transforms.CenterCrop(280),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("[*] Test verisi yükleniyor...")
    test_dataset = Medical25DDataset(root_dir=test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"[*] Toplam test edilecek hasta/görüntü sayısı: {len(test_dataset)}")

    model = build_model(num_classes=2).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"[+] En iyi model ağırlıkları başarıyla yüklendi: {os.path.basename(model_path)}")
    else:
        print("[-] HATA: Model dosyası bulunamadı!")
        exit()

    model.eval() 

    y_true = []
    y_pred = []

    print("[>>>] TEST BAŞLIYOR [<<<]")
    with torch.no_grad(): 
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # ==========================================
    # 4. SONUÇLAR VE GRAFİK
    # ==========================================
    sınıf_isimleri = ['Normal (Sağlıklı)', 'Abnormal (Hasta)']
    
    print("\n" + "="*50)
    print("KLİNİK SINIFLANDIRMA RAPORU")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=sınıf_isimleri))
    print("="*50)

    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6), facecolor='white')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sınıf_isimleri, 
                yticklabels=sınıf_isimleri,
                annot_kws={"size": 16})
                
    plt.title('Test Verisi Karmaşıklık Matrisi (Dengeli Model)', fontsize=14)
    plt.xlabel('Modelin Tahmini', fontsize=12)
    plt.ylabel('Gerçek Durum', fontsize=12)
    plt.tight_layout()

    grafik_yolu = os.path.join(rapor_klasoru, "test_karmasiklik_matrisi_balanced.png")
    plt.savefig(grafik_yolu, dpi=300)
    print(f"\n[+] Matris grafiği kaydedildi: {grafik_yolu}")
    
    plt.show()