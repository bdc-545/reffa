import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import timm
import pandas as pd
from tqdm import tqdm

from PIL import Image
import re

# =========================
# CONFIG
# =========================
DATA_DIR = r"D:\bdc_satriadata_2026\BDC2026\BDC_2026\Data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

BATCH_SIZE = 16
EPOCHS = 10
LR = 3e-4
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "efficientnet_b0"   # ganti ke b1 kalau mau lebih kuat

# =========================
# TRANSFORM (ANTI OVERFITTING)
# =========================
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# =========================
# DATASET
# =========================
dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
val_data.dataset.transform = val_tf

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# MODEL (EfficientNet)
# =========================
model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=3
).to(DEVICE)

# =========================
# LOSS + OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# =========================
# EVALUATION (MACRO F1)
# =========================
def evaluate(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            pred = torch.argmax(out, dim=1)

            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())

    return f1_score(labels, preds, average='macro')

# =========================
# TRAINING LOOP
# =========================
best_f1 = 0
patience = 0

for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for x, y in loop:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    scheduler.step()

    val_f1 = evaluate(model, val_loader)
    print(f"\nEpoch {epoch+1} | Val Macro F1: {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best_effnet.pth")
        print("Saved Best Model!")
        patience = 0
    else:
        patience += 1

    if patience >= 3:
        print("Early stopping triggered")
        break

# =========================
# INFERENCE
# =========================
print("Loading best model...")
model.load_state_dict(torch.load("best_effnet.pth"))
model.eval()

# Fungsi pendukung urutan angka natural (1, 2, 3... bukan 1, 10, 11)
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# Baca isi folder test langsung dan urutkan secara natural
test_files = sorted(
    [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
    key=natural_sort_key
)

predictions = []
id_gambar = [] # Tempat menampung nama ID dari file

with torch.no_grad():
    for file_name in tqdm(test_files, desc="Menebak Data Test"):
        file_path = os.path.join(TEST_DIR, file_name)

        img = Image.open(file_path).convert('RGB')
        x = val_tf(img).unsqueeze(0).to(DEVICE)

        out = model(x)
        pred = torch.argmax(out, dim=1)

        predictions.append(pred.item())

        # Ambil nama file asli tanpa .jpg (misal '1.jpg' -> '1') untuk ID
        nama_tanpa_ext = os.path.splitext(file_name)[0]
        id_gambar.append(nama_tanpa_ext)

# =========================
# SUBMISSION
# =========================
df = pd.DataFrame({
    # ID benar-benar mengambil dari nama asli gambar yang sedang ditebak
    "id": id_gambar,
    "predicted": predictions
})

df.to_csv("submission.csv", index=False)
print("submission.csv created!")