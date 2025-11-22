import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ============================================================
# 1. CONFIGURATION (ตั้งค่า)
# ============================================================
# 🔥 ตรวจสอบ Path ให้ถูกต้อง
DATA_DIR = r"C:\Users\rutsa\PycharmProjects\MLproject\MLproject\dataset"
MODEL_SAVE_PATH = 'dog_model_pytorch.pth'

IMG_SIZE = 224
BATCH_SIZE = 16  # ลดลงหน่อยเพื่อให้ Fine-tuning ได้ละเอียดขึ้น
EPOCHS = 20  # เพิ่มรอบการเรียนรู้
LEARNING_RATE = 1e-4  # 🔥 สำคัญ: ใช้ LR ต่ำๆ เพื่อค่อยๆ จูน (Fine-tuning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ============================================================
# 2. DATA AUGMENTATION (เพิ่มความยากให้โจทย์)
# ============================================================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # สุ่มพลิกภาพ/หมุนเล็กน้อย (อย่าหมุนเยอะ เดี๋ยวโมเดลงงเงา)
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        # 🔥 เพิ่มการปรับสี/แสง เพื่อให้โมเดลไม่จำแค่สี แต่ดู texture
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# ============================================================
# 3. PROCESS
# ============================================================
def load_data():
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms['train'])
    class_names = full_dataset.classes
    print(f"✅ Classes: {class_names}")

    # เช็คว่าเรียงลำดับถูกไหม (0=ai, 1=real ปกติโฟลเดอร์จะเรียงตามตัวอักษร)
    if class_names[0] != 'ai':
        print("⚠️ Warning: Class order might be unexpected. Check folder names.")

    # Split Data (70/15/15)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # เปลี่ยน Transform ของ Val/Test ให้เป็นแบบนิ่งๆ (ไม่มี Random)
    val_dataset.dataset.transform = data_transforms['val']
    test_dataset.dataset.transform = data_transforms['val']

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        'test': DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }

    return dataloaders, class_names, len(train_dataset), len(val_dataset)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # เก็บโมเดลที่ดีที่สุดไว้ (ไม่ใช่โมเดลรอบสุดท้าย)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Loop ผ่านข้อมูล
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase}"):
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = (torch.sigmoid(outputs) > 0.5).float()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                # ปรับ Learning Rate ถ้า loss ไม่ลดลง
                scheduler.step(epoch_loss)

                # 🔥 Save Best Model Only
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f"⭐ Found better model! (Acc: {best_acc:.4f})")

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # คืนค่าโมเดลที่เป็นร่างทอง (Best Weights)
    model.load_state_dict(best_model_wts)
    return model


# ============================================================
# 4. MAIN
# ============================================================
if __name__ == '__main__':
    dataloaders, class_names, train_size, val_size = load_data()
    dataset_sizes = {'train': train_size, 'val': val_size}

    print("🛠️  Building EfficientNet B0 (Fine-Tuning Mode)...")

    # โหลดโมเดลพื้นฐาน
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # 🔥 KEY CHANGE 1: Unfreeze (เปิดให้เรียนรู้ทุกชั้น)
    for param in model.parameters():
        param.requires_grad = True

        # เปลี่ยน Layer สุดท้าย
    num_ftrs = model.classifier[1].in_features
    # เพิ่ม Dropout เพื่อกัน Overfitting
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 1)
    )

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    # 🔥 KEY CHANGE 2: Optimizer & Scheduler
    # ใช้ LR ต่ำๆ (1e-4) เพราะเรา Unfreeze โมเดล ถ้าสูงไปความรู้เก่าจะพัง
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # ลด Learning Rate ลงถ้าระบบเริ่มตัน (ช่วยให้จูนแม่นขึ้นตอนท้าย)
    # ลบ verbose=True ออก
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # Train
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=EPOCHS)

    # Save
    print(f"💾 Saving BEST model to {MODEL_SAVE_PATH}...")
    torch.save(model, MODEL_SAVE_PATH)
    print("✅ Done! Ready to deploy.")