import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as tv_transforms  
import monai.transforms as mt  

# Kendi yazdığın modüller
from Monai import Medical3DDataset, build_monai_model

def main():
    # ==========================================
    # 1. HİPERPARAMETRELER VE YOLLAR
    # ==========================================
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(base_dir, "proje_verisi_yeni", "proje_verisi_yeni")
    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "val") 
    save_path = os.path.join(base_dir, "kaydedilen_modeller", "best_monai_densenet3d_augmented.pth")
    
    BATCH_SIZE = 4
    EPOCHS = 30
    LEARNING_RATE = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Kullanılan Donanım: {device}")

    # ==========================================
    # 2. DÖNÜŞÜMLER (İKİ AŞAMALI YAPI)
    # ==========================================
    # AŞAMA 1: Medical3DDataset'in içindeki PIL Image dilimlerine uygulanacak 2D dönüşümler
    slice_transforms = tv_transforms.Compose([
        tv_transforms.Resize((224, 224)),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # AŞAMA 2: Eğitim döngüsünde 3D hacim oluştuktan sonra uygulanacak MONAI 3D dönüşümleri
    volume_aug_transforms = mt.Compose([
        mt.RandRotate90(prob=0.5, spatial_axes=(1, 2)), 
        mt.RandFlip(prob=0.5, spatial_axis=1),          
        mt.RandAffine(
            prob=0.3,
            rotate_range=(0.1, 0.1, 0.1),
            translate_range=(5, 5, 5),
            scale_range=(0.1, 0.1, 0.1)
        )
    ])

    # ==========================================
    # 3. VERİ YÜKLEYİCİLER (DATALOADERS)
    # ==========================================
    print("[*] Eğitim ve Validasyon veri setleri yükleniyor...")
    # PIL hatası almamak için Dataset'e sadece 2D dönüşümleri veriyoruz
    train_dataset = Medical3DDataset(root_dir=train_dir, depth=64, transform=slice_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    if os.path.exists(val_dir):
        val_dataset = Medical3DDataset(root_dir=val_dir, depth=64, transform=slice_transforms)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    else:
        val_loader = None
        print("[!] Validasyon klasörü bulunamadı, sadece eğitim yapılacak.")

    # ==========================================
    # 4. MODEL, LOSS VE OPTİMİZATÖR
    # ==========================================
    model = build_monai_model(num_classes=2).to(device)

    # Azınlık sınıfı (Anormal) korumak için ağırlıklandırma (False Negative azaltma odaklı)
    # 250 Normal (0), 199 Anormal (1)
    weight_normal = 449 / (2 * 250) 
    weight_abnormal = 449 / (2 * 199) 
    class_weights = torch.tensor([weight_normal, weight_abnormal], dtype=torch.float).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # ==========================================
    # 5. EĞİTİM DÖNGÜSÜ (TRAINING LOOP)
    # ==========================================
    print("\n[>>>] EĞİTİM BAŞLIYOR [<<<]")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            # inputs boyutu şu an: [Batch, 1, 64, 224, 224] (Saf 3D Tensor)
            
            # --- 3D DATA AUGMENTATION VE METATENSOR ÇÖZÜMÜ ---
            augmented_inputs = []
            for i in range(inputs.size(0)): 
                aug_vol = volume_aug_transforms(inputs[i]) 
                augmented_inputs.append(aug_vol)
            
            # 1. Dönüştürülmüş MR'ları birleştir
            inputs = torch.stack(augmented_inputs)
            
            # 2. MONAI'nin MetaTensor kılıfını çıkarıp saf PyTorch Tensor'üne çevir (.item() hatasını çözer)
            if hasattr(inputs, "as_tensor"):
                inputs = inputs.as_tensor()
                
            # 3. Donanıma gönder
            inputs = inputs.to(device)
            # ------------------------------------------------
            
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Artık saf Tensor olduğu için .item() metodu çökmez
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_train
        epoch_acc = correct_train / total_train
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f}")

        # Validasyon Aşaması
        if val_loader:
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    # Validasyonda rastgele döndürme UYGULANMAZ
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            avg_val_loss = val_loss / total_val
            val_acc = correct_val / total_val
            print(f"       -> Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f}")

            # En iyi modeli kaydet
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"       [*] Model iyileşti, kaydedildi: {save_path}")

    # Validasyon yoksa son durumu kaydet
    if not val_loader:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"\n[*] Eğitim tamamlandı. Model kaydedildi: {save_path}")

if __name__ == '__main__':
    main()