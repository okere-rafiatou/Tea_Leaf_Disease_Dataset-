import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from baseline.model import build_model

# ── Config ──────────────────────────────────────────────
TEST_DIR        = "data/test"
WEIGHTS_PATH    = "model_weights.pth"
OUTPUT_FILE     = "submissions/submission.csv"
IMAGE_SIZE      = 224
BATCH_SIZE      = 32
NUM_CLASSES     = 8
CLASS_NAMES     = ['Anthracnose', 'algal leaf', 'bird eye spot',
                   'brown blight', 'gray light', 'healthy',
                   'red leaf spot', 'white spot']
# ────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir   = img_dir
        self.transform = transform
        self.filenames = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path  = os.path.join(self.img_dir, self.filenames[idx])
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.filenames[idx]


test_ds     = TestDataset(TEST_DIR, transform=transform)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

model = build_model(NUM_CLASSES)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model = model.to(device)
model.eval()

os.makedirs("submissions", exist_ok=True)

all_preds = []
with torch.no_grad():
    for images, filenames in test_loader:
        images  = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for pred in predicted.cpu().numpy():
            all_preds.append(CLASS_NAMES[pred])

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    for label in all_preds:
        writer.writerow([label])

print(f"Saved {len(all_preds)} predictions to {OUTPUT_FILE}")
