"""
Demo W&B: Experiment Tracking + Media Logging + Sweeps
Dataset: CIFAR-10 | Model: Simple CNN (PyTorch)

Kịch bản cover:
  - Phần 2: log metrics real-time (loss, accuracy mỗi epoch)
  - Phần 3: log ảnh dự đoán sai (media logging)
  - Phần 4: Sweep với Bayesian optimization

Cách chạy:
  # Chạy thường (1 run):
  python train_wandb.py

  # Chạy Sweep:
  wandb sweep sweep_config.yaml
  wandb agent <SWEEP_ID>
"""

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

CLASSES = ("plane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

# ── Model ────────────────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ── Data ─────────────────────────────────────────────────────────────────────

def get_loaders(batch_size=64):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=2)
    return train_loader, test_loader

# ── Log ảnh dự đoán sai (Phần 3 demo) ───────────────────────────────────────

def log_wrong_predictions(model, loader, device, n=16):
    """Log những ảnh model dự đoán sai lên W&B."""
    model.eval()
    wrong_images = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            mask = preds != labels
            for img, pred, label in zip(
                    imgs[mask], preds[mask], labels[mask]):
                if len(wrong_images) >= n:
                    break
                # Denormalize để hiển thị
                mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
                std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)
                img_show = (img.cpu() * std + mean).clamp(0, 1)
                caption = (f"Pred: {CLASSES[pred.item()]} | "
                           f"True: {CLASSES[label.item()]}")
                wrong_images.append(
                    wandb.Image(img_show.permute(1,2,0).numpy(),
                                caption=caption))
            if len(wrong_images) >= n:
                break
    wandb.log({"wrong_predictions": wrong_images})

# ── Train / Eval ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            correct += out.argmax(1).eq(labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Khởi tạo run — W&B tự tạo project nếu chưa có
    run = wandb.init(
        project="cifar10-demo",
        config={
            "learning_rate": 1e-3,
            "batch_size": 64,
            "epochs": 5,
            "dropout": 0.3,
            "optimizer": "adam",
        },
    )
    cfg = wandb.config  # Sweep sẽ override các giá trị này

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_loaders(cfg.batch_size)
    model = SimpleCNN(dropout=cfg.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Theo dõi gradients & kiến trúc model (Phần 2 demo)
    wandb.watch(model, log="gradients", log_freq=100)

    best_val_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(
            model, test_loader, criterion, device)
        scheduler.step()

        # ── Log metrics real-time (Phần 2 demo) ──────────────────────────
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "lr": scheduler.get_last_lr()[0],
        })

        print(f"[{epoch:02d}/{cfg.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        # Lưu model tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

    # ── Log ảnh dự đoán sai (Phần 3 demo) ────────────────────────────────
    log_wrong_predictions(model, test_loader, device)

    # ── Log model artifact (Phần 3 demo) ─────────────────────────────────
    artifact = wandb.Artifact("cifar10-cnn", type="model",
                              description="Best CNN checkpoint")
    artifact.add_file("best_model.pt")
    run.log_artifact(artifact)

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    wandb.summary["best_val_accuracy"] = best_val_acc
    run.finish()


if __name__ == "__main__":
    main()