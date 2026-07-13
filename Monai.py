import os
import glob
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# MONAI Kütüphaneleri
import monai
from monai.networks.nets import DenseNet121
from monai.transforms import Resize

# ==========================================
# 1. VERİ SETİ SINIFI (MONAI & MR Uyumu İçin 1-Kanal)
# ==========================================
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

class Medical3DDataset(Dataset):
    def __init__(self, root_dir, depth=64, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['normal', 'abnormal']
        self.depth = depth # 3D Hacim için alınacak ardışık kesit sayısı
        self.samples = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for sample_folder in os.listdir(class_dir):
                sample_path = os.path.join(class_dir, sample_folder)
                if os.path.isdir(sample_path):
                    png_files = glob.glob(os.path.join(sample_path, "*.png"))
                    # Sadece yeterli kesite sahip hacimleri eğitime dahil et
                    if len(png_files) >= self.depth: 
                        self.samples.append((sample_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, label = self.samples[idx]
        
        image_files = glob.glob(os.path.join(sample_path, "*.png"))
        image_files.sort(key=natural_sort_key)

        # Hacmin tam ortasından ardışık DEPTH (16) kadar kesiti al
        start_idx = (len(image_files) - self.depth) // 2
        selected_files = image_files[start_idx : start_idx + self.depth]

        frames = []
        for img_path in selected_files:
            # DİKKAT: MR verisi olduğu için RGB yerine 'L' (Grayscale - 1 Kanal) kullanıyoruz.
            # Bu sayede VRAM kullanımını 3'te 1 oranında azaltıyoruz!
            img = Image.open(img_path).convert('L')
            
            if self.transform:
                img_tensor = self.transform(img) # [1, H, W]
            else:
                img_tensor = transforms.ToTensor()(img)
                
            frames.append(img_tensor)

        # 16 adet [1, 224, 224] kesiti Derinlik (dim=1) ekseninde birleştiriyoruz.
        # Sonuç: [1, 16, 224, 224] formatında tam bir medikal 3D matris. (Channel, Depth, Height, Width)
        volume_3d = torch.stack(frames, dim=1) 

        return volume_3d, label

# ==========================================
# 2. MODEL MİMARİSİ (MONAI 3D DenseNet)
# ==========================================
def build_monai_model(num_classes=2):
    # MONAI'nin 3D tıbbi veriler için optimize edilmiş DenseNet121 modeli
    # spatial_dims=3 (3D veri), in_channels=1 (Gri tonlamalı MR)
    model = DenseNet121(
        spatial_dims=3,
        in_channels=1, 
        out_channels=num_classes,
        dropout_prob=0.3  # Aşırı öğrenmeyi (overfitting) engellemek için
    )
    return model

# ==========================================
# 3. ANA ÇALIŞTIRMA VE EĞİTİM DÖNGÜSÜ
# ==========================================
if __name__ == '__main__':
    # ---------------------------------------------------------
    # DOSYA YOLLARI
    # ---------------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(base_dir, "proje_verisi_yeni", "proje_verisi_yeni")

    train_dir = os.path.join(dataset_root, "train")
    val_dir   = os.path.join(dataset_root, "val")
    test_dir  = os.path.join(dataset_root, "test")
    output_dir = os.path.join(base_dir, "kaydedilen_modeller")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir) or not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Veri klasörleri bulunamadı. Beklenen konum: {dataset_root}"
        )

    # Veriyi 1 kanala düşürdüğümüz için bellekte (VRAM) daha fazla yer açıldı.
    # Eğer sisteminiz destekliyorsa BATCH_SIZE'ı 8'e bile çıkarabilirsiniz.
    BATCH_SIZE = 4 
    EPOCHS = 10
    LEARNING_RATE = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Kullanılan Donanım: {device}")

    # Görüntü Dönüşümleri (1 kanal için uygun hale getirildi)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # Tek kanal için basit normalizasyon (-1 ile 1 arasına çeker)
    ])

    print("[*] 3D MR Veri setleri (Volume) yükleniyor...")
    train_dataset = Medical3DDataset(root_dir=train_dir, depth=64, transform=data_transforms)
    val_dataset = Medical3DDataset(root_dir=val_dir, depth=64, transform=data_transforms)
    test_dataset = Medical3DDataset(root_dir=test_dir, depth=64, transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"[*] Eğitim: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)} hacim")

    model = build_monai_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    # Tıbbi modellerde AdamW kullanmak ağırlık sönümlemesi (weight decay) ile genellikle daha iyi sonuç verir
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    use_amp = True if device.type == 'cuda' else False
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_val_loss = float('inf')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # MONAI model ismiyle kaydediyoruz
    model_save_path = os.path.join(output_dir, "best_monai_densenet3d.pth")

    print("\n[>>>] MONAI 3D DENSENET EĞİTİMİ BAŞLIYOR [<<<]")
    for epoch in range(EPOCHS):
        
        # --- EĞİTİM FAZI ---
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        avg_train_loss = train_loss / total_train
        train_acc = 100 * correct_train / total_train
        
        # --- DOĞRULAMA (VALIDATION) FAZI ---
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        avg_val_loss = val_loss / total_val
        val_acc = 100 * correct_val / total_val
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} Acc: %{train_acc:.2f} | Val Loss: {avg_val_loss:.4f} Acc: %{val_acc:.2f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  [+] Yeni en iyi MONAI modeli kaydedildi! (Val Loss: {best_val_loss:.4f})")

    # ==========================================
    # 4. FİNAL TESTİ
    # ==========================================
    print("\n[>>>] EĞİTİM BİTTİ. MONAI MODELİ TEST EDİLİYOR [<<<]")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    correct_test, total_test = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
    final_test_acc = 100 * correct_test / total_test
    print(f"[!] MONAI DENSENET MODELİ TEST VERİSİ BAŞARISI: %{final_test_acc:.2f}")