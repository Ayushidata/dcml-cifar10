"""
==============================================================
EXPERIMENT A — MODEL-CENTRIC APPROACH
==============================================================
Goal    : Keep data FIXED (raw CIFAR-10), improve performance
          by trying different model architectures and tuning
          hyperparameters.
Models  : ResNet-18, VGG-16, MobileNet-V2
Dataset : CIFAR-10 (raw, no cleaning)
Metrics : Accuracy, F1-Score (macro), Confusion Matrix
Seeds   : 5 seeds for statistical reliability
==============================================================
"""

import os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, vgg16, mobilenet_v2
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR    = "./data"
RESULTS_DIR = "./results"
NUM_CLASSES = 10
EPOCHS      = 15
SEEDS       = [42, 123, 256, 512, 999]   # FIX: 5 seeds for mean±std

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

HYPERPARAM_GRID = {
    "ResNet18":    [(0.01, 64),  (0.001, 128)],
    "VGG16":       [(0.001, 32), (0.0005, 64)],
    "MobileNetV2": [(0.01, 64),  (0.001, 128)],
}

os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Device: {DEVICE}")


def get_loaders(batch_size=64):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = torchvision.datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=tf)
    test_set  = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256,        shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def build_model(name):
    if name == "ResNet18":
        m = resnet18(weights="IMAGENET1K_V1")
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    elif name == "VGG16":
        m = vgg16(weights="IMAGENET1K_V1")
        m.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    elif name == "MobileNetV2":
        m = mobilenet_v2(weights="IMAGENET1K_V1")
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m.to(DEVICE)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += out.argmax(1).eq(lbls).sum().item()
        total      += lbls.size(0)
    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            out = model(imgs.to(DEVICE))
            preds_all.extend(out.argmax(1).cpu().numpy())
            labels_all.extend(lbls.numpy())
    preds_all  = np.array(preds_all)
    labels_all = np.array(labels_all)
    acc = 100.0 * (preds_all == labels_all).mean()
    f1  = f1_score(labels_all, preds_all, average="macro") * 100
    return acc, f1, preds_all, labels_all


def run_single_seed(model_name, lr, batch_size, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, test_loader = get_loaders(batch_size)
    model     = build_model(model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc, best_f1 = 0.0, 0.0
    best_preds = best_labels = None

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        te_acc, te_f1, preds, labels = evaluate(model, test_loader)
        scheduler.step()
        if te_acc > best_acc:
            best_acc, best_f1 = te_acc, te_f1
            best_preds, best_labels = preds, labels
        print(f"  [seed={seed}] Ep {epoch:02d}/{EPOCHS} | "
              f"loss={tr_loss:.4f} | train={tr_acc:.1f}% | "
              f"test={te_acc:.2f}% | f1={te_f1:.2f}%")

    return best_acc, best_f1, best_preds, best_labels


def run(model_name, lr, batch_size):
    print(f"\n{'='*55}\n  {model_name} | lr={lr} | batch={batch_size}\n{'='*55}")
    seed_accs, seed_f1s = [], []
    last_preds = last_labels = None

    for seed in SEEDS:
        acc, f1, preds, labels = run_single_seed(model_name, lr, batch_size, seed)
        seed_accs.append(acc)
        seed_f1s.append(f1)
        last_preds, last_labels = preds, labels

    mean_acc = float(np.mean(seed_accs))
    std_acc  = float(np.std(seed_accs))
    mean_f1  = float(np.mean(seed_f1s))
    std_f1   = float(np.std(seed_f1s))

    print(f"\n  BEST acc={mean_acc:.2f} ± {std_acc:.2f}%  "
          f"f1={mean_f1:.2f} ± {std_f1:.2f}%")

    save_cm(last_labels, last_preds,
            f"Exp A — {model_name} lr={lr}",
            f"{RESULTS_DIR}/cm_A_{model_name}_lr{lr}.png")

    return dict(
        model=model_name, lr=lr, batch_size=batch_size,
        mean_acc=round(mean_acc, 2), std_acc=round(std_acc, 2),
        mean_f1=round(mean_f1, 2),   std_f1=round(std_f1, 2),
        seed_accs=seed_accs, seed_f1s=seed_f1s
    )


def save_cm(labels, preds, title, fname):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CIFAR10_CLASSES,
                yticklabels=CIFAR10_CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()
    print(f"  Saved: {fname}")


def plot_summary(all_results):
    models = [f"{r['model']}\nlr={r['lr']}" for r in all_results]
    accs   = [r["mean_acc"] for r in all_results]
    stds   = [r["std_acc"]  for r in all_results]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, accs, yerr=stds, capsize=5, color="#4C72B0", alpha=0.85, label="Accuracy ± std")
    for i, (a, s) in enumerate(zip(accs, stds)):
        ax.text(i, a + s + 0.5, f"{a:.1f}±{s:.2f}", ha="center", fontsize=8)
    ax.set_title("Experiment A — Model-Centric Results (mean ± std, n=5 seeds)")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("Test Accuracy (%)"); ax.set_ylim(0, 105)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = f"{RESULTS_DIR}/summary_A.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"Saved: {p}")


def main():
    print("\n" + "="*55)
    print("  EXPERIMENT A — MODEL-CENTRIC APPROACH")
    print(f"  Seeds: {SEEDS}")
    print("="*55)

    all_results = []
    for model_name, configs in HYPERPARAM_GRID.items():
        for lr, bs in configs:
            all_results.append(run(model_name, lr, bs))

    # Save JSON
    with open(f"{RESULTS_DIR}/results_A.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results_A.json")

    plot_summary(all_results)

    print("\n" + "="*55)
    print("  LEADERBOARD — EXPERIMENT A (mean ± std)")
    print(f"  {'Model':<14}{'LR':<9}{'Acc%':<20}{'F1%'}")
    print("  " + "-"*55)
    for r in sorted(all_results, key=lambda x: -x["mean_acc"]):
        print(f"  {r['model']:<14}{r['lr']:<9}"
              f"{r['mean_acc']:.2f} ± {r['std_acc']:.2f}    "
              f"{r['mean_f1']:.2f} ± {r['std_f1']:.2f}")


if __name__ == "__main__":
    main()
