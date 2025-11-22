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
from tqdm import tqdm  # สำหรับ Progress Bar สวยๆ

# ============================================================
# 1. CONFIGURATION
# ============================================================
DATA_DIR = r"C:\Users\rutsa\PycharmProjects\MLproject\MLproject\dataset"  # แก้ path ให้ตรง
MODEL_SAVE_PATH = 'dog_model_pytorch.pth'
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# ตรวจสอบ GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
if device.type == 'cuda':
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# 2. DATA PREPARATION (Transforms)
# ============================================================
# PyTorch ต้องการ Normalization ตามมาตรฐาน ImageNet
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
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
# 3. LOAD DATASET
# ============================================================
def load_data():
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms['train'])
    class_names = full_dataset.classes
    print(f"✅ Classes Found: {class_names}")  # ควรเป็น ['ai', 'real']

    # Split Data (Train 70%, Val 15%, Test 15%)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply 'val' transform to val/test datasets (เพื่อไม่ให้มี Data Augmentation ตอนเทส)
    val_dataset.dataset.transform = data_transforms['val']
    test_dataset.dataset.transform = data_transforms['val']

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        # num_workers ปรับตาม CPU cores
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4),
        'test': DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }

    return dataloaders, class_names, len(train_dataset), len(val_dataset)


# ============================================================
# 4. TRAINING FUNCTION
# ============================================================
def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

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

            # Iterate over data (with Progress Bar)
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} phase"):
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)  # แปลง label เป็น shape [batch, 1]

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # outputs เป็น Logits ต้องผ่าน Sigmoid ถ้าจะดูค่าจริง แต่ BCEWithLogitsLoss รับ Logits ได้เลย
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

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"🌟 New Best Validation Accuracy: {best_acc:.4f}")

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history


# ============================================================
# 5. MAIN EXECUTION
# ============================================================
if __name__ == '__main__':
    # 1. Load Data
    try:
        dataloaders, class_names, train_size, val_size = load_data()
        dataset_sizes = {'train': train_size, 'val': val_size}
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        exit()

    # 2. Setup Model (EfficientNet B0)
    print("🛠️  Building EfficientNet B0 Model...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze weights (Optional: ถ้าข้อมูลน้อยให้ Freeze ไว้ก่อน)
    for param in model.features.parameters():
        param.requires_grad = False

        # แก้ไข Output Layer สำหรับ Binary Classification (1 node)
    # EfficientNet B0 output สุดท้ายอยู่ที่ classifier[1]
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1)

    model = model.to(device)

    # 3. Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()  # เหมาะกับ Binary Classification มากกว่า MSE
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Train
    model, history = train_model(model, criterion, optimizer, num_epochs=EPOCHS)

    # 5. Save Model
    print(f"💾 Saving model to {MODEL_SAVE_PATH}...")
    torch.save(model, MODEL_SAVE_PATH)  # เซฟทั้งโมเดล (Structure + Weights)
    print("✅ Model saved successfully!")

    # 6. Evaluation on Test Set & Confusion Matrix
    print("\n📝 Evaluating on Test Set...")
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy().flatten()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    plt.figure(figsize=(6, 6))
    disp.plot(cmap="Blues", ax=plt.gca())
    plt.title("Confusion Matrix (PyTorch)")
    plt.show()