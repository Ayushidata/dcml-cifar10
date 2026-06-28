"""
==============================================================
EXPERIMENT C — HYBRID PIPELINE (RESEARCH CONTRIBUTION)
==============================================================
Goal    : Combine DCAI data pipeline + best model from Exp A

Hybrid Pipeline:
  Phase 1 — DATA: CleanLab + Dedup + Augmentation
  Phase 2 — MODEL: VGG-16 + Mixup + LR Warmup + Label Smoothing
  Phase 3 — COMPARISON: A vs B vs C

Seeds   : 5 seeds for statistical reliability (mean ± std)
==============================================================
"""

import os, time, json, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, vgg16, mobilenet_v2
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from cleanlab.filter import find_label_issues
    CLEANLAB_AVAILABLE = True
except ImportError:
    CLEANLAB_AVAILABLE = False

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR      = "./data"
RESULTS_DIR   = "./results"
NUM_CLASSES   = 10
EPOCHS        = 15
NOISE_RATE    = 0.20
SEEDS         = [42, 123, 256, 512, 999]   # 5 seeds for mean±std
WARMUP_EPOCHS = 3
BASE_LR       = 0.001
BATCH_SIZE    = 32

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Device: {DEVICE}")


# ── DATA PIPELINE ─────────────────────────────────────────────────────────────

def load_cifar10_numpy():
    raw    = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True)
    return raw.data, np.array(raw.targets)


def inject_noise(labels, rate, seed=42):
    rng   = np.random.default_rng(seed)
    noisy = labels.copy()
    idx   = rng.choice(len(labels), size=int(len(labels)*rate), replace=False)
    for i in idx:
        choices  = [c for c in range(NUM_CLASSES) if c != noisy[i]]
        noisy[i] = rng.choice(choices)
    mask = noisy != labels
    print(f"  Injected {mask.sum()} noisy labels ({100*mask.mean():.1f}%)")
    return noisy, mask


def get_oof_probs(images_np, labels_np, n_splits=5):
    from sklearn.model_selection import StratifiedKFold
    N    = len(labels_np)
    prob = np.zeros((N, NUM_CLASSES), dtype=np.float32)
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1)
    std  = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1)
    print(f"  Computing {n_splits}-fold OOF probabilities...")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(images_np, labels_np)):
        print(f"    Fold {fold+1}/{n_splits}...", end=" ", flush=True)
        t0    = time.time()
        X_tr  = (torch.from_numpy(images_np[tr_idx]).permute(0,3,1,2).float()/255 - mean)/std
        y_tr  = torch.from_numpy(labels_np[tr_idx]).long()
        X_val = (torch.from_numpy(images_np[val_idx]).permute(0,3,1,2).float()/255 - mean)/std
        m     = resnet18(weights=None)
        m.fc  = nn.Linear(m.fc.in_features, NUM_CLASSES)
        m     = m.to(DEVICE)
        opt   = optim.Adam(m.parameters(), lr=1e-3)
        crit  = nn.CrossEntropyLoss()
        ld    = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)
        for _ in range(3):
            m.train()
            for xb, yb in ld:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
        m.eval()
        vld = DataLoader(TensorDataset(X_val, torch.zeros(len(X_val)).long()), batch_size=256)
        probs_fold = []
        with torch.no_grad():
            for xb, _ in vld:
                probs_fold.append(torch.softmax(m(xb.to(DEVICE)),1).cpu().numpy())
        prob[val_idx] = np.vstack(probs_fold)
        print(f"{time.time()-t0:.1f}s")
    return prob


def clean_labels(images, noisy_labels, noise_mask):
    probs = get_oof_probs(images, noisy_labels)
    if CLEANLAB_AVAILABLE:
        issue_idx = find_label_issues(
            labels=noisy_labels, pred_probs=probs,
            return_indices_ranked_by="self_confidence")
        detected = np.zeros(len(noisy_labels), dtype=bool)
        detected[issue_idx] = True
    else:
        detected = probs[np.arange(len(noisy_labels)), noisy_labels] < 0.5
    tp = (detected & noise_mask).sum()
    if detected.sum() > 0:
        print(f"  Recall={100*tp/noise_mask.sum():.1f}%  "
              f"Precision={100*tp/detected.sum():.1f}%  "
              f"Kept={(~detected).sum()}")
    return images[~detected], noisy_labels[~detected]


def deduplicate(images, labels):
    seen, keep = {}, []
    for i, img in enumerate(images):
        h = hashlib.md5(img.tobytes()).hexdigest()
        if h not in seen:
            seen[h] = True; keep.append(i)
    keep = np.array(keep)
    print(f"  Dedup: removed {len(images)-len(keep)}, kept {len(keep)}")
    return images[keep], labels[keep]


class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, lbls, tf):
        self.imgs=imgs; self.lbls=lbls; self.tf=tf
    def __len__(self): return len(self.lbls)
    def __getitem__(self, i): return self.tf(self.imgs[i]), int(self.lbls[i])


def make_train_loader(images, labels):
    mean = (0.4914, 0.4822, 0.4465); std = (0.2023, 0.1994, 0.2010)
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return DataLoader(NumpyDataset(images, labels, tf),
                      batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=2, pin_memory=True)


def get_test_loader():
    mean = (0.4914, 0.4822, 0.4465); std = (0.2023, 0.1994, 0.2010)
    tf   = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    ts   = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf)
    return DataLoader(ts, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)


# ── MODEL PIPELINE ────────────────────────────────────────────────────────────

def build_model(name):
    if name == "VGG16":
        m = vgg16(weights="IMAGENET1K_V1")
        m.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    elif name == "ResNet18":
        m = resnet18(weights="IMAGENET1K_V1")
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    elif name == "MobileNetV2":
        m = mobilenet_v2(weights="IMAGENET1K_V1")
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown: {name}")
    return m.to(DEVICE)


def mixup_data(x, y, alpha=0.4):
    lam  = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx  = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam


def mixup_criterion(crit, pred, ya, yb, lam):
    return lam*crit(pred, ya) + (1-lam)*crit(pred, yb)


def get_lr(epoch, warmup, base_lr, total):
    if epoch < warmup:
        return base_lr * (epoch+1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


def train_one_seed(train_loader, test_loader, model_name, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m    = build_model(model_name)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt  = optim.SGD(m.parameters(), lr=BASE_LR, momentum=0.9, weight_decay=5e-4)

    best_acc, best_f1 = 0.0, 0.0
    best_preds = best_labels = None

    for epoch in range(EPOCHS):
        lr = get_lr(epoch, WARMUP_EPOCHS, BASE_LR, EPOCHS)
        for pg in opt.param_groups: pg["lr"] = lr

        m.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            imgs, ya, yb, lam = mixup_data(imgs, lbls)
            opt.zero_grad()
            mixup_criterion(crit, m(imgs), ya, yb, lam).backward()
            opt.step()

        m.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for imgs, lbls in test_loader:
                preds_all.extend(m(imgs.to(DEVICE)).argmax(1).cpu().numpy())
                labels_all.extend(lbls.numpy())
        pa = np.array(preds_all); la = np.array(labels_all)
        acc = 100.0 * (pa == la).mean()
        f1  = f1_score(la, pa, average="macro") * 100
        if acc > best_acc:
            best_acc, best_f1 = acc, f1
            best_preds, best_labels = pa, la
        print(f"  [seed={seed}] Ep {epoch+1:02d} | lr={lr:.5f} | "
              f"test={acc:.2f}% | f1={f1:.2f}%")

    return best_acc, best_f1, best_preds, best_labels


def save_cm(labels, preds, title, fname):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=CIFAR10_CLASSES,
                yticklabels=CIFAR10_CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()
    print(f"  Saved: {fname}")


def plot_final_comparison(all_results):
    labels = list(all_results.keys())
    accs   = [all_results[k]["mean_acc"] for k in labels]
    stds   = [all_results[k].get("std_acc", 0) for k in labels]
    colors = ["#4C72B0", "#2196F3", "#E74C3C"]
    x      = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, accs, yerr=stds, capsize=6,
                  color=colors[:len(labels)], alpha=0.85)
    for i, (a, s) in enumerate(zip(accs, stds)):
        ax.text(i, a + s + 0.5, f"{a:.2f}±{s:.2f}%",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Final Comparison — Exp A vs B vs C (mean ± std, n=5 seeds)",
                 fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Test Accuracy (%)"); ax.set_ylim(60, 100)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = f"{RESULTS_DIR}/FINAL_COMPARISON_ABC.png"
    plt.savefig(p, dpi=180); plt.close()
    print(f"Saved: {p}")


def main():
    print("\n" + "="*60)
    print("  EXPERIMENT C — HYBRID PIPELINE")
    print(f"  Seeds: {SEEDS}  |  BASE_LR={BASE_LR}  |  Warmup={WARMUP_EPOCHS} epochs")
    print("="*60)

    # ── Phase 1: Data pipeline ────────────────────────────────
    print("\n[PHASE 1] DATA PIPELINE")
    images, clean_labels = load_cifar10_numpy()
    noisy_labels, noise_mask = inject_noise(clean_labels, NOISE_RATE, seed=42)
    images_c, labels_c = clean_labels(images, noisy_labels, noise_mask)
    images_c, labels_c = deduplicate(images_c, labels_c)
    print(f"\n  Final training set: {len(images_c)} samples")

    train_loader = make_train_loader(images_c, labels_c)
    test_loader  = get_test_loader()

    # ── Phase 2: Model — read best from Exp A ────────────────
    print("\n[PHASE 2] MODEL PIPELINE")
    best_model_name = "VGG16"
    res_A_path = f"{RESULTS_DIR}/results_A.json"
    if os.path.exists(res_A_path):
        with open(res_A_path) as f:
            res_A = json.load(f)
        best_A = max(res_A, key=lambda x: x.get("mean_acc", x.get("best_acc", 0)))
        best_model_name = best_A["model"]
        acc_key = "mean_acc" if "mean_acc" in best_A else "best_acc"
        print(f"  Best model from Exp A: {best_model_name} ({best_A[acc_key]:.2f}%)")
    else:
        print(f"  Exp A results not found — using default: {best_model_name}")

    accs, f1s = [], []
    last_preds = last_labels = None
    for seed in SEEDS:
        acc, f1, preds, lbls = train_one_seed(
            train_loader, test_loader, best_model_name, seed)
        accs.append(acc); f1s.append(f1)
        last_preds, last_labels = preds, lbls

    mean_acc = float(np.mean(accs)); std_acc = float(np.std(accs))
    mean_f1  = float(np.mean(f1s));  std_f1  = float(np.std(f1s))
    print(f"\n  HYBRID: acc={mean_acc:.2f} ± {std_acc:.2f}%  "
          f"f1={mean_f1:.2f} ± {std_f1:.2f}%")

    save_cm(last_labels, last_preds,
            "Experiment C — Hybrid Pipeline",
            f"{RESULTS_DIR}/cm_C_hybrid.png")

    # ── Phase 3: Final comparison ─────────────────────────────
    print("\n[PHASE 3] FINAL COMPARISON")
    all_results = {}

    if os.path.exists(f"{RESULTS_DIR}/results_A.json"):
        with open(f"{RESULTS_DIR}/results_A.json") as f:
            res_A = json.load(f)
        best_A = max(res_A, key=lambda x: x.get("mean_acc", x.get("best_acc", 0)))
        acc_k  = "mean_acc" if "mean_acc" in best_A else "best_acc"
        std_k  = "std_acc"  if "std_acc"  in best_A else None
        all_results[f"Exp A (MCAI)\n{best_A['model']}"] = {
            "mean_acc": best_A[acc_k],
            "std_acc":  best_A[std_k] if std_k else 0.0
        }

    if os.path.exists(f"{RESULTS_DIR}/results_B.json"):
        with open(f"{RESULTS_DIR}/results_B.json") as f:
            res_B = json.load(f)
        b_std = res_B.get("final_std", 0.0)
        all_results["Exp B (DCAI)\nResNet-18"] = {
            "mean_acc": res_B["final_acc"],
            "std_acc":  b_std
        }

    all_results["Exp C (Hybrid)\n" + best_model_name] = {
        "mean_acc": mean_acc, "std_acc": std_acc
    }

    with open(f"{RESULTS_DIR}/results_C.json", "w") as f:
        json.dump({
            "model":      best_model_name,
            "mean_acc":   round(mean_acc, 2), "std_acc": round(std_acc, 2),
            "mean_f1":    round(mean_f1, 2),  "std_f1":  round(std_f1, 2),
            "seed_accs":  accs, "seed_f1s": f1s,
            "seeds":      SEEDS,
            "all_results": {k: v for k, v in all_results.items()},
        }, f, indent=2)

    plot_final_comparison(all_results)

    print("\n" + "="*65)
    print("  MASTER RESULTS TABLE (for research paper)")
    print(f"  {'Experiment':<25}{'Approach':<20}{'Acc%'}")
    print("  " + "-"*65)
    for name, vals in all_results.items():
        label = name.split("\n")[0]
        approach = ("Model-Centric" if "MCAI" in name
                    else "Data-Centric" if "DCAI" in name
                    else "Hybrid (Ours)")
        print(f"  {label:<25}{approach:<20}"
              f"{vals['mean_acc']:.2f} ± {vals['std_acc']:.2f}%")


if __name__ == "__main__":
    main()
