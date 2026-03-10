import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from baseline.model import build_model

# ── Config ──────────────────────────────────────────────
DATA_DIR   = "data/train"
SAVE_PATH  = "model_weights.pth"
BATCH_SIZE = 32
EPOCHS     = 20
LR         = 1e-4
IMAGE_SIZE = 224
NUM_CLASSES = 8
# ────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(root=DATA_DIR)
targets      = np.array([label for _, label in full_dataset.imgs])

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

train_ds = Subset(datasets.ImageFolder(root=DATA_DIR, transform=train_transform), train_idx)
val_ds   = Subset(datasets.ImageFolder(root=DATA_DIR, transform=val_transform),   val_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
print(f"Classes: {full_dataset.classes}")

model     = build_model(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    # ── Train ──
    model.train()
    running_loss = correct = total = 0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted  = torch.max(outputs, 1)
        correct       += (predicted == labels).sum().item()
        total         += labels.size(0)
        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

    train_acc = 100. * correct / total
    print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%")

    # ── Validate ──
    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total   += labels.size(0)

    val_acc = 100. * val_correct / val_total
    print(f"Epoch {epoch+1}: Val   Acc={val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  Saved best model at epoch {epoch+1} ({val_acc:.2f}%)")

    scheduler.step()

print(f"\nTraining complete. Best val accuracy: {best_val_acc:.2f}%")
print(f"Model saved to {SAVE_PATH}")
