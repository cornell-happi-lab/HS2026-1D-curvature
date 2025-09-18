# train_strong.py
import os, argparse, json
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

NODE_COLS = [f"node{i}" for i in range(1,17)]

# ---------- Args ----------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="sensor_data.csv",
                    help="Raw data with node1..node16 and kappa (+ optional sensor_id).")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--mixup_alpha", type=float, default=0.15)
    ap.add_argument("--label_jitter", type=float, default=0.01)  # on scaled targets
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="runs_strong")
    return ap.parse_args()

# ---------- Robust loader ----------
def load_table(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df.shape[1] < 3: raise ValueError
        return df
    except Exception:
        return pd.read_csv(path, sep=r"[,\t ]+", engine="python")

# ---------- Feature functions (must be reused in realtime) ----------
def engineered_features_from_z(z: np.ndarray) -> np.ndarray:
    """
    z: (B,16) z-scored node matrix
    returns: (B,8) engineered features computed on z
    """
    # robust aggregations on z
    sum_z   = z.sum(axis=1, keepdims=True)
    mean_z  = z.mean(axis=1, keepdims=True)
    std_z   = z.std(axis=1, keepdims=True)
    min_z   = z.min(axis=1, keepdims=True)
    max_z   = z.max(axis=1, keepdims=True)
    range_z = max_z - min_z
    l2norm  = np.linalg.norm(z, axis=1, keepdims=True)
    q75     = np.quantile(z, 0.75, axis=1, keepdims=True)
    q25     = np.quantile(z, 0.25, axis=1, keepdims=True)
    iqr     = (q75 - q25)
    feats = np.concatenate([sum_z, mean_z, std_z, min_z, max_z, range_z, l2norm, iqr], axis=1)
    return feats.astype(np.float32)

# ---------- Dataset ----------
class SensorDataset(Dataset):
    def __init__(self, df, node_mean=None, node_std=None, feat_mean=None, feat_std=None, fit_stats=False):
        X_nodes = df[NODE_COLS].to_numpy(dtype=np.float32)
        y = df["kappa"].to_numpy(dtype=np.float32).reshape(-1,1)  # 0..80
        # Fit/Apply node z-score on TRAIN only
        if fit_stats:
            node_mean = X_nodes.mean(axis=0, keepdims=True)
            node_std  = X_nodes.std(axis=0, keepdims=True) + 1e-8
        z = (X_nodes - node_mean) / node_std
        # Engineered features on z
        feats = engineered_features_from_z(z)
        if fit_stats:
            feat_mean = feats.mean(axis=0, keepdims=True)
            feat_std  = feats.std(axis=0, keepdims=True) + 1e-8
        feats_z = (feats - feat_mean) / feat_std
        X = np.concatenate([z, feats_z], axis=1)  # shape: (B, 16+8)

        self.X = X.astype(np.float32)
        self.y = (y / 100.0).astype(np.float32)   # scale targets to 0..0.8

        self.node_mean = node_mean
        self.node_std  = node_std
        self.feat_mean = feat_mean
        self.feat_std  = feat_std

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

# ---------- Model (Residual MLP) ----------
class ResBlock(nn.Module):
    def __init__(self, d, p=0.10):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(d, d), nn.SiLU(), nn.Dropout(p=p),
            nn.Linear(d, d)
        )
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(x + self.seq(x))

class StrongMLP(nn.Module):
    def __init__(self, in_dim=24, dropout=0.10):
        super().__init__()
        width = 128
        self.stem = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Dropout(p=dropout)
        )
        self.blocks = nn.Sequential(ResBlock(width, dropout),
                                    ResBlock(width, dropout),
                                    ResBlock(width, dropout))
        self.head = nn.Sequential(
            nn.Linear(width, 64), nn.SiLU(), nn.Dropout(p=dropout),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)

# ---------- Mixup ----------
def mixup(X, y, alpha=0.15):
    if alpha is None or alpha <= 0: return X, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(X.size(0))
    return lam * X + (1 - lam) * X[idx], lam * y + (1 - lam) * y[idx]

# ---------- Splitting ----------
def smart_split(df, val_ratio, test_ratio, seed):
    N = len(df)
    if "sensor_id" in df.columns:
        # Group-aware split (keeps entire sensors out of test)
        gss1 = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        train_val_idx, test_idx = next(gss1.split(np.zeros(N), groups=df["sensor_id"].values))
        tv = df.iloc[train_val_idx].reset_index(drop=True)
        te = df.iloc[test_idx].reset_index(drop=True)

        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio / (1 - test_ratio), random_state=seed+1)
        train_idx, val_idx = next(gss2.split(np.zeros(len(tv)), groups=tv["sensor_id"].values))
        tr = tv.iloc[train_idx].reset_index(drop=True)
        va = tv.iloc[val_idx].reset_index(drop=True)
    else:
        # Stratify by kappa if no sensor_id
        y = df["kappa"].astype(int).to_numpy()
        n_test = int(round(test_ratio * N))
        n_val  = int(round(val_ratio  * N))
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=seed)
        train_val_idx, test_idx = next(sss1.split(np.zeros(N), y))
        tv = df.iloc[train_val_idx].reset_index(drop=True)
        te = df.iloc[test_idx].reset_index(drop=True)

        y_tv = tv["kappa"].astype(int).to_numpy()
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=n_val, random_state=seed+1)
        train_idx, val_idx = next(sss2.split(np.zeros(len(tv)), y_tv))
        tr = tv.iloc[train_idx].reset_index(drop=True)
        va = tv.iloc[val_idx].reset_index(drop=True)
    return tr, va, te

# ---------- Train/Eval ----------
def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    df = load_table(args.csv)
    # Basic sanity & casting
    for c in NODE_COLS + ["kappa"]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {args.csv}")
    df = df.copy()
    for c in NODE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["kappa"] = pd.to_numeric(df["kappa"], errors="coerce").astype(int)
    df = df.dropna(subset=NODE_COLS + ["kappa"]).reset_index(drop=True)

    # Split
    train_df, val_df, test_df = smart_split(df, args.val_ratio, args.test_ratio, args.seed)

    # Datasets (fit z-score on train only)
    train_set = SensorDataset(train_df, fit_stats=True)
    val_set   = SensorDataset(val_df,
                              node_mean=train_set.node_mean, node_std=train_set.node_std,
                              feat_mean=train_set.feat_mean, feat_std=train_set.feat_std)
    test_set  = SensorDataset(test_df,
                              node_mean=train_set.node_mean, node_std=train_set.node_std,
                              feat_mean=train_set.feat_mean, feat_std=train_set.feat_std)

    trL = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    vaL = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False)
    teL = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False)

    # Model / loss / optim / scheduler
    model = StrongMLP(in_dim=train_set.X.shape[1], dropout=args.dropout)
    criterion = nn.HuberLoss(delta=0.02)  # robust to outliers on scaled targets
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    best_val = float("inf")
    best_path = os.path.join(args.outdir, "best.pt")
    patience = 999; bad = 0

    for ep in range(1, args.epochs+1):
        # ---- train ----
        model.train(); tr_loss = 0.0
        for xb, yb in trL:
            xb, yb = mixup(xb, yb, alpha=args.mixup_alpha)
            if args.label_jitter > 0:
                yb = yb + args.label_jitter * torch.randn_like(yb)

            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward(); opt.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(trL))

        # ---- val ----
        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for xb, yb in vaL:
                va_loss += criterion(model(xb), yb).item()
        va_loss /= max(1, len(vaL))
        scheduler.step()

        if va_loss < best_val:
            best_val, bad = va_loss, 0
            torch.save({
                "model": model.state_dict(),
                "node_mean": torch.from_numpy(train_set.node_mean).float(),
                "node_std":  torch.from_numpy(train_set.node_std).float(),
                "feat_mean": torch.from_numpy(train_set.feat_mean).float(),
                "feat_std":  torch.from_numpy(train_set.feat_std).float(),
                "in_dim":    int(train_set.X.shape[1]),
                "feature_spec": {"engineered": ["sum","mean","std","min","max","range","l2","iqr"]},
            }, best_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {ep} (best val {best_val:.6f})")
                break

        if ep % 10 == 0 or ep == 1:
            print(f"Epoch {ep:03d} | train {tr_loss:.5f} | val {va_loss:.5f}")

    # ---- Test (report in 1/m) ----
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in teL:
            p = model(xb).cpu().numpy() * 100.0
            t = yb.cpu().numpy() * 100.0
            preds.append(p); trues.append(t)
    preds = np.vstack(preds).ravel()
    trues = np.vstack(trues).ravel()

    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae  = mean_absolute_error(trues, preds)
    r2   = r2_score(trues, preds)
    print(f"\nTEST (1/m): RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.4f}")

    # Quick scatter
    plt.figure(figsize=(5,5))
    plt.scatter(trues, preds, s=16)
    lo, hi = min(trues.min(), preds.min()), max(trues.max(), preds.max())
    plt.plot([lo,hi],[lo,hi],"--")
    plt.xlabel("True κ (1/m)"); plt.ylabel("Pred κ (1/m)")
    plt.title("Pred vs True")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
