import json
import random
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score
import os

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Same EDA stats as before for Normalization
stats = json.load(open("outputs/eda_summary.json"))
MEAN, STD = stats["train_mean"], stats["train_std"]
IMG = 128

train_tfm = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_tfm = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Load dataset from folders
full_dataset = datasets.ImageFolder('dataset', transform=None)
print("Classes:", full_dataset.classes) # Should be ['avalanche', 'no_avalanche']

# We need to manually apply different transforms to train and test, so we split indices
num_samples = len(full_dataset)
indices = list(range(num_samples))
random.shuffle(indices)

train_size = int(0.7 * num_samples)
val_size = int(0.15 * num_samples)

train_idx = indices[:train_size]
val_idx = indices[train_size:train_size+val_size]
test_idx = indices[train_size+val_size:]

class SubsetWithTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

train_data = SubsetWithTransform(Subset(full_dataset, train_idx), train_tfm)
val_data = SubsetWithTransform(Subset(full_dataset, val_idx), eval_tfm)
test_data = SubsetWithTransform(Subset(full_dataset, test_idx), eval_tfm)

loaders = {
    "train": DataLoader(train_data, batch_size=8, shuffle=True),
    "val": DataLoader(val_data, batch_size=8, shuffle=False),
    "test": DataLoader(test_data, batch_size=8, shuffle=False),
}

# Same CNN Architecture
class CNN(nn.Module):
    def __init__(self, hidden=128, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 4 * 4, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2), # 2 classes: avalanche, no_avalanche
        )
    def forward(self, x):
        return self.head(self.features(x))

def evaluate(model, loader):
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            y_pred += model(xb).argmax(dim=1).cpu().tolist()
            y_true += yb.tolist()
    acc = 100.0 * sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)
    return acc, y_pred, y_true

def train_model(hidden=256, dropout=0.5, lr=0.001, wd=1e-4, max_epochs=25, patience=6):
    model = CNN(hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    best = {"f1": -1.0, "state": None}
    bad = 0
    
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            
        _, yp, yt = evaluate(model, loaders["val"])
        f1 = f1_score(yt, yp, average="macro", zero_division=0)
        
        if f1 > best["f1"]:
            best = {"f1": f1, "state": {k: v.cpu() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= patience: break
            
    model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
    return model

print("Training new avalanche detection model...")
model = train_model(hidden=256, dropout=0.5, lr=0.001)

# Evaluate on test
acc, yp, yt = evaluate(model, loaders["test"])
print(f"Test Accuracy: {acc:.2f}%")

# Save new model
torch.save(model.state_dict(), "outputs/avalanche_cnn.pt")
print("Saved outputs/avalanche_cnn.pt")

# Save results config
payload = {
    "best_config": {"lr": 0.001, "hidden": 256, "dropout": 0.5, "wd": 1e-4},
    "test_accuracy": acc,
    "classes": full_dataset.classes
}
json.dump(payload, open("outputs/avalanche_results.json", "w"), indent=2)
