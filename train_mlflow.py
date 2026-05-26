"""
Demo MLflow: Experiment Tracking + Model Registry + Serving
Dataset: CIFAR-10 | Model: Simple CNN (PyTorch)

Kịch bản cover:
  - Phần 2: log metrics (loss, accuracy mỗi epoch)
  - Phần 3: log model artifact + đăng ký lên Model Registry
  - Phần 5: serve model (xem hướng dẫn cuối file)

Cách chạy:
  # Khởi động MLflow UI (terminal riêng):
  mlflow ui --port 5000

  # Chạy training:
  python train_mlflow.py

  # Sau khi chạy xong, vào http://localhost:5000 để xem runs
  # Đăng ký model lên Registry (demo trực tiếp trên UI hoặc dùng lệnh bên dưới)

  # Serve model (Phần 5 demo):
  mlflow models serve -m "models:/cifar10-cnn/Production" -p 5001 --no-conda
"""

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

CLASSES = ("plane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

# ── Model (giống train_wandb.py để so sánh công bằng) ────────────────────────

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
    # Hyperparameters
    config = {
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 5,
        "dropout": 0.3,
        "optimizer": "adam",
    }

    # Trỏ MLflow đến server local (mặc định ./mlruns nếu không set)
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("cifar10-demo")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_loaders(config["batch_size"])

    with mlflow.start_run(run_name="cnn-baseline") as run:
        print(f"Run ID: {run.info.run_id}")

        # ── Log hyperparameters (Phần 2 demo) ────────────────────────────
        mlflow.log_params(config)
        mlflow.set_tag("model_type", "SimpleCNN")
        mlflow.set_tag("dataset", "CIFAR-10")

        model = SimpleCNN(dropout=config["dropout"]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(1, config["epochs"] + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = eval_epoch(
                model, test_loader, criterion, device)
            scheduler.step()

            # ── Log metrics theo step (Phần 2 demo) ──────────────────────
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)

            print(f"[{epoch:02d}/{config['epochs']}] "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()}

        # Load lại best weights trước khi log model
        model.load_state_dict(best_model_state)
        mlflow.log_metric("best_val_accuracy", best_val_acc)

        # ── Log model artifact (Phần 3 demo) ─────────────────────────────
        # Move model về CPU trước khi log để tránh cpu/cuda mismatch
        sample_input = torch.randn(1, 3, 32, 32, dtype=torch.float32)
        mlflow.pytorch.log_model(
            pytorch_model=model.cpu(),
            artifact_path="model",
            registered_model_name="cifar10-cnn",   # tự đăng ký lên Registry
            input_example=sample_input.numpy(),
            serialization_format="pt2",            # tránh warning pickle
        )

        print(f"\nBest val accuracy: {best_val_acc:.3f}")
        print("\n--- Bước tiếp theo (demo trên UI) ---")
        print("1. Mở http://localhost:5000")
        print("2. Vào Models → cifar10-cnn → chuyển stage sang Production")
        print("3. Serve: mlflow models serve -m 'models:/cifar10-cnn/Production'"
              " -p 5001 --no-conda")
        print("4. Gọi thử:")
        print('   curl -X POST http://localhost:5001/invocations \\')
        print('        -H "Content-Type: application/json" \\')
        print('        -d \'{"inputs": [[...]]}\' ')


# ── Hàm test serving (Phần 5 demo) ───────────────────────────────────────────

def test_serving():
    """
    Gọi REST API sau khi đã serve model.
    Chạy sau khi `mlflow models serve` đã khởi động ở port 5001.
    """
    import requests, json

    # Tạo ảnh ngẫu nhiên giả (1 ảnh CIFAR-10: 3x32x32)
    dummy = np.random.rand(1, 3, 32, 32).tolist()
    payload = json.dumps({"inputs": dummy})

    resp = requests.post(
        "http://localhost:5001/invocations",
        headers={"Content-Type": "application/json"},
        data=payload,
    )
    logits = resp.json()["predictions"][0]
    pred_class = CLASSES[int(np.argmax(logits))]
    print(f"Predicted class: {pred_class}")
    print(f"Raw logits: {[round(x, 3) for x in logits]}")


if __name__ == "__main__":
    main()
    # Bỏ comment dòng dưới để test serving sau khi model đã được serve:
    # test_serving()