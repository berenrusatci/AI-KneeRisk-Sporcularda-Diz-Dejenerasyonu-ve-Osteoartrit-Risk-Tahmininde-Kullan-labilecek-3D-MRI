import os
import glob
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# 1. DİNAMİK VERİ SETİ SINIFI (2.5D)
# ==========================================
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

class Medical25DDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
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
        
        mid_idx = len(image_files) // 2
        
        if self.is_train and len(image_files) > 7:
            offset = random.randint(-2, 2)
            mid_idx = max(1, min(len(image_files) - 2, mid_idx + offset))

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
# 2. İLERİ SEVİYE KAYIP FONKSİYONU (FOCAL LOSS)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0): 
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma 

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

# ==========================================
# 3. MODEL MİMARİSİ
# ==========================================
def build_model(num_classes=2):
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False
            
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.5),  
        nn.Linear(num_ftrs, num_classes)
    )
    return model

# ==========================================
# 4. ANA ÇALIŞTIRMA VE EĞİTİM DÖNGÜSÜ
# ==========================================
if __name__ == '__main__':
    train_dir = r"C:\Projeler\tbtk2209a\proje_verisi_yeni\train"
    val_dir   = r"C:\Projeler\tbtk2209a\proje_verisi_yeni\val"
    test_dir  = r"C:\Projeler\tbtk2209a\proje_verisi_yeni\test"
    output_dir = r"C:\Projeler\tbtk2209a\kaydedilen_modeller"

    BATCH_SIZE = 16 
    EPOCHS_PHASE_1 = 10 
    EPOCHS_PHASE_2 = 40 
    EARLY_STOP_PATIENCE = 8 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Kullanılan Donanım: {device}")

    # ==========================================
    # YENİLİK 1: Gelişmiş Ön İşleme ve Kırpma
    # ==========================================
    train_transforms = transforms.Compose([
        transforms.CenterCrop(280), # 1. Adım: Kenardaki siyahlıkları ve cihaz yazılarını kes
        transforms.Resize((224, 224), antialias=True), # 2. Adım: Sadece diz eklemini yeniden boyutlandır
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)), # 3. Adım: Görüntüye rastgele siyah kutular koyarak ezberi boz
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.CenterCrop(280), # Doğrulama için de önce kırp
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = Medical25DDataset(root_dir=train_dir, transform=train_transforms, is_train=True)
    val_dataset = Medical25DDataset(root_dir=val_dir, transform=val_test_transforms, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model(num_classes=2).to(device)
    
    # ==========================================
    # YENİLİK 2: Ağırlık Dengelemesi
    # ==========================================
    # Eski ağırlık [1.0, 1.6] idi. Şimdi evhamı azaltmak için hastalık ağırlığını 1.25'e düşürdük.
    weights = torch.tensor([1.0, 1.25]).to(device)
    criterion = FocalLoss(weight=weights, gamma=2.0)
    
    use_amp = True if device.type == 'cuda' else False
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Yeni modelin adını değiştiriyoruz ki eskisiyle karışmasın
    model_save_path = os.path.join(output_dir, "best_model_balanced_crop.pth")

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')

    print(f"\n[>>>] FAZ 1: ISINMA BAŞLIYOR ({EPOCHS_PHASE_1} Epoch) [<<<]")
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-2)
    
    for epoch in range(EPOCHS_PHASE_1):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        avg_train_loss = train_loss / total_train
        train_acc = 100 * correct_train / total_train
        
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
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"FAZ 1 - Epoch [{epoch+1}/{EPOCHS_PHASE_1}] | Train Loss: {avg_train_loss:.4f} Acc: %{train_acc:.2f} | Val Loss: {avg_val_loss:.4f} Acc: %{val_acc:.2f}")

    print(f"\n[>>>] FAZ 2: İNCE AYAR BAŞLIYOR ({EPOCHS_PHASE_2} Epoch) [<<<]")
    for param in model.parameters():
        param.requires_grad = True 
        
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE_2, eta_min=1e-7)

    epochs_trained_phase2 = 0
    epochs_no_improve = 0

    for epoch in range(EPOCHS_PHASE_2):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        avg_train_loss = train_loss / total_train
        train_acc = 100 * correct_train / total_train
        
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
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        epochs_trained_phase2 += 1
        
        print(f"FAZ 2 - Epoch [{epoch+1}/{EPOCHS_PHASE_2}] [LR: {current_lr:.2e}] | Train Loss: {avg_train_loss:.4f} Acc: %{train_acc:.2f} | Val Loss: {avg_val_loss:.4f} Acc: %{val_acc:.2f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  [+] Yeni en iyi dengeli model kaydedildi! (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n[!] Doğrulama hatası {EARLY_STOP_PATIENCE} turdur düşmüyor. Erken durdurma tetiklendi!")
            break

    print("\n[*] Eğitim tamamlandı. Öğrenme eğrileri çizdiriliyor...")
    
    total_trained_epochs = EPOCHS_PHASE_1 + epochs_trained_phase2
    epochs_range = range(1, total_trained_epochs + 1)
    
    plt.figure(figsize=(14, 5), facecolor='white', edgecolor='black', linewidth=2)

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(epochs_range, history['train_loss'], label='Eğitim Loss', color='blue', linewidth=2)
    ax1.plot(epochs_range, history['val_loss'], label='Doğrulama Loss', color='red', linewidth=2)
    ax1.axvline(x=EPOCHS_PHASE_1, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Model Hata (Loss) Eğrisi')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(epochs_range, history['train_acc'], label='Eğitim Başarısı', color='blue', linewidth=2)
    ax2.plot(epochs_range, history['val_acc'], label='Doğrulama Başarısı', color='red', linewidth=2)
    ax2.axvline(x=EPOCHS_PHASE_1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('Model Doğruluk (Accuracy) Eğrisi')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    grafik_yolu = r"C:\Projeler\tbtk2209a\ogrenme_egrileri_balanced.png"
    plt.savefig(grafik_yolu, dpi=300, bbox_inches='tight')
    plt.show()