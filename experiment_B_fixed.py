"""
==============================================================
EXPERIMENT B — DATA-CENTRIC APPROACH
==============================================================
Goal    : Keep model FIXED (ResNet-18), improve performance
          by systematically improving the DATASET quality.

Pipeline:
  B0 — Inject 20% synthetic label noise
  B1 — Detect & remove noisy labels (Confident Learning)
  B2 — Remove duplicate images (MD5 hashing)
  B3 — Apply data augmentation

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
from torchvision.models import resnet18
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from cleanlab.filter import find_label_issues
    CLEANLAB_AVAILABLE = True
except ImportError:
    print("CleanLab not installed. Run: pip install cleanlab")
    CLEANLAB_AVAILABLE = False

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR    = "./data"
RESULTS_DIR = "./results"
NUM_CLASSES = 10
EPOCHS      = 15
NOISE_RATE  = 0.20
SEEDS       = [42, 123, 256, 512, 999]   # FIX: 5 seeds for mean±std
LR          = 0.01                        # FIX: was 0.001, correct is 0.01
BATCH_SIZE  = 64

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Device: {DEVICE}")


def load_cifar10_numpy():
    raw    = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True)
    images = raw.data
    labels = np.array(raw.targets)
    return images, labels


def inject_label_noise(labels, noise_rate, seed):
    rng   = np.random.default_rng(seed)
    noisy = labels.copy()
    n_noisy = int(len(labels) * noise_rate)
    noisy_indices = rng.choice(len(labels), size=n_noisy, replace=False)
    for idx in noisy_indices:
        choices  = [c for c in range(NUM_CLASSES) if c != noisy[idx]]
        noisy[idx] = rng.choice(choices)
    noise_mask = noisy != labels
    print(f"  Injected {noise_mask.sum()} noisy labels ({100*noise_mask.mean():.1f}%)")
    return noisy, noise_mask


def get_oof_probs(images_np, labels_np, n_splits=5):
    from sklearn.model_selection import StratifiedKFold
    N = len(labels_np)
    prob_matrix = np.zeros((N, NUM_CLASSES), dtype=np.float32)
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1)
    std  = torch.tensor((0.2023, 0.1994, 0.2010)).view(3, 1, 1)

    print(f"  Computing {n_splits}-fold OOF probabilities for CleanLab...")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(images_np, labels_np)):
        print(f"    Fold {fold+1}/{n_splits}...", end=" ", flush=True)
        t0 = time.time()
        X_tr  = (torch.from_numpy(images_np[tr_idx]).permute(0,3,1,2).float()/255 - mean)/std
        y_tr  = torch.from_numpy(labels_np[tr_idx]).long()
        X_val = (torch.from_numpy(images_np[val_idx]).permute(0,3,1,2).float()/255 - mean)/std

        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        m = m.to(DEVICE)
        opt  = optim.Adam(m.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        tr_ld = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)

        for _ in range(3):
            m.train()
            for xb, yb in tr_ld:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()

        m.eval()
        val_ld = DataLoader(TensorDataset(X_val, torch.zeros(len(X_val)).long()), batch_size=256)
        probs  = []
        with torch.no_grad():
            for xb, _ in val_ld:
                probs.append(torch.softmax(m(xb.to(DEVICE)), 1).cpu().numpy())
        prob_matrix[val_idx] = np.vstack(probs)
        print(f"{time.time()-t0:.1f}s")
    return prob_matrix


def detect_and_clean(images, noisy_labels, noise_mask):
    probs = get_oof_probs(images, noisy_labels)
    if CLEANLAB_AVAILABLE:
        issue_idx = find_label_issues(
            labels=noisy_labels, pred_probs=probs,
            return_indices_ranked_by="self_confidence"
        )
        detected = np.zeros(len(noisy_labels), dtype=bool)
        detected[issue_idx] = True
    else:
        self_conf = probs[np.arange(len(noisy_labels)), noisy_labels]
        detected  = self_conf < 0.5

    tp = (detected & noise_mask).sum()
    fp = (detected & ~noise_mask).sum()
    recall    = tp / noise_mask.sum() * 100
    precision = tp / detected.sum() * 100 if detected.sum() > 0 else 0
    print(f"  Recall={recall:.1f}%  Precision={precision:.1f}%  "
          f"Removed={detected.sum()}  Kept={len(images)-detected.sum()}")
    return images[~detected], noisy_labels[~detected], recall, precision


def remove_duplicates(images, labels):
    seen, keep = {}, []
    for i, img in enumerate(images):
        h = hashlib.md5(img.tobytes()).hexdigest()
        if h not in seen:
            seen[h] = True; keep.append(i)
    keep = np.array(keep)
    print(f"  Dedup: removed {len(images)-len(keep)}, kept {len(keep)}")
    return images[keep], labels[keep]


class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, lbls, transform):
        self.imgs = imgs; self.lbls = lbls; self.tf = transform
    def __len__(self): return len(self.lbls)
    def __getitem__(self, i):
        return self.tf(self.imgs[i]), int(self.lbls[i])


def make_loader(images, labels, augment=False):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    if augment:
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return DataLoader(NumpyDataset(images, labels, tf),
                      batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=2, pin_memory=True)


def get_test_loader():
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    tf   = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    ts   = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf)
    return DataLoader(ts, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)


def train_one_seed(train_loader, test_loader, seed):
    torch.manual_seed(seed)
    m     = resnet18(weights="IMAGENET1K_V1")
    m.fc  = nn.Linear(m.fc.in_features, NUM_CLASSES)
    m     = m.to(DEVICE)
    crit  = nn.CrossEntropyLoss()
    opt   = optim.SGD(m.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_acc, best_f1 = 0.0, 0.0
    best_preds = best_labels = None

    for epoch in range(1, EPOCHS + 1):
        m.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad()
            loss = crit(m(imgs), lbls)
            loss.backward(); opt.step()
        sched.step()
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
        print(f"  [seed={seed}] Ep {epoch:02d} | test={acc:.2f}% | f1={f1:.2f}%")

    return best_acc, best_f1, best_preds, best_labels


def run_stage(images, labels, augment, stage_name, test_loader):
    loader = make_loader(images, labels, augment=augment)
    accs, f1s = [], []
    last_preds = last_labels = None
    for seed in SEEDS:
        acc, f1, preds, lbls = train_one_seed(loader, test_loader, seed)
        accs.append(acc); f1s.append(f1)
        last_preds, last_labels = preds, lbls
    mean_acc = float(np.mean(accs)); std_acc = float(np.std(accs))
    mean_f1  = float(np.mean(f1s));  std_f1  = float(np.std(f1s))
    print(f"\n  [{stage_name}] acc={mean_acc:.2f}±{std_acc:.2f}%  "
          f"f1={mean_f1:.2f}±{std_f1:.2f}%\n")
    return mean_acc, std_acc, mean_f1, std_f1, last_preds, last_labels


def save_cm(labels, preds, title, fname):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=CIFAR10_CLASSES,
                yticklabels=CIFAR10_CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()


def plot_comparison(results):
    stages = list(results.keys())
    accs   = [results[s]["mean_acc"] for s in stages]
    stds   = [results[s]["std_acc"]  for s in stages]
    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, accs, yerr=stds, capsize=5, color="#2196F3", alpha=0.85)
    for i, (a, s) in enumerate(zip(accs, stds)):
        ax.text(i, a + s + 0.3, f"{a:.2f}±{s:.2f}", ha="center", fontsize=8)
    ax.set_title("Experiment B — Data-Centric Stages (mean ± std, n=5 seeds)")
    ax.set_xticks(x); ax.set_xticklabels(stages, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(60, 85)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/comparison_B.png", dpi=150); plt.close()
    print(f"Saved: {RESULTS_DIR}/comparison_B.png")


def main():
    print("\n" + "="*60)
    print("  EXPERIMENT B — DATA-CENTRIC APPROACH")
    print(f"  Model: ResNet-18 fixed | LR={LR} | Seeds: {SEEDS}")
    print("="*60)

    test_loader = get_test_loader()
    images, clean_labels = load_cifar10_numpy()
    results = {}

    # B0: Noisy baseline — use seed 42 for noise injection (reproducible)
    print("\n[B0] Noisy baseline")
    noisy_labels, noise_mask = inject_label_noise(clean_labels, NOISE_RATE, seed=42)
    acc, std, f1, sf1, preds, lbls = run_stage(
        images, noisy_labels, False, "B0-Noisy", test_loader)
    results["B0: Noisy (baseline)"] = dict(mean_acc=acc, std_acc=std, mean_f1=f1, std_f1=sf1)
    save_cm(lbls, preds, "Exp B0 — Noisy Baseline",
            f"{RESULTS_DIR}/cm_B0_noisy.png")

    # B1: Label cleaning
    print("\n[B1] Label cleaning — Confident Learning")
    images_c, labels_c, recall, precision = detect_and_clean(
        images, noisy_labels, noise_mask)
    acc, std, f1, sf1, preds, lbls = run_stage(
        images_c, labels_c, False, "B1-Cleaned", test_loader)
    results["B1: Clean labels"] = dict(mean_acc=acc, std_acc=std, mean_f1=f1, std_f1=sf1,
                                        recall=round(recall,1), precision=round(precision,1))
    save_cm(lbls, preds, "Exp B1 — After Label Cleaning",
            f"{RESULTS_DIR}/cm_B1_cleaned.png")

    # B2: Deduplication
    print("\n[B2] Deduplication")
    images_dd, labels_dd = remove_duplicates(images_c, labels_c)
    acc, std, f1, sf1, preds, lbls = run_stage(
        images_dd, labels_dd, False, "B2-Dedup", test_loader)
    results["B2: + Dedup"] = dict(mean_acc=acc, std_acc=std, mean_f1=f1, std_f1=sf1)
    save_cm(lbls, preds, "Exp B2 — After Deduplication",
            f"{RESULTS_DIR}/cm_B2_dedup.png")

    # B3: Augmentation
    print("\n[B3] Augmentation — final DCAI result")
    acc, std, f1, sf1, preds, lbls = run_stage(
        images_dd, labels_dd, True, "B3-Augmented", test_loader)
    results["B3: + Augmentation"] = dict(mean_acc=acc, std_acc=std, mean_f1=f1, std_f1=sf1)
    save_cm(lbls, preds, "Exp B — Final Data-Centric Result",
            f"{RESULTS_DIR}/cm_B3_final.png")

    with open(f"{RESULTS_DIR}/results_B.json", "w") as f:
        json.dump({
            "stages":         results,
            "final_acc":      results["B3: + Augmentation"]["mean_acc"],
            "final_std":      results["B3: + Augmentation"]["std_acc"],
            "final_f1":       results["B3: + Augmentation"]["mean_f1"],
            "noise_detection":{"recall": recall, "precision": precision},
            "seeds":          SEEDS,
            "lr":             LR,
        }, f, indent=2)

    plot_comparison(results)

    print("\n" + "="*60)
    print("  EXPERIMENT B — SUMMARY")
    print(f"  {'Stage':<25}{'Acc%':<22}{'F1%'}")
    print("  " + "-"*60)
    baseline = results["B0: Noisy (baseline)"]["mean_acc"]
    for stage, v in results.items():
        delta = f"  (+{v['mean_acc']-baseline:.2f}pp)" if stage != "B0: Noisy (baseline)" else ""
        print(f"  {stage:<25}{v['mean_acc']:.2f} ± {v['std_acc']:.2f}%     "
              f"{v['mean_f1']:.2f}{delta}")


if __name__ == "__main__":
    main()
