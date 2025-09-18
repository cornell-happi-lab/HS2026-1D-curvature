# train_cylinder_curvature_mixup.py
import argparse, os, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ---------- Args ----------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="sensor_data.csv",
                    help="Path to data file. Supports comma or whitespace-separated.")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--mixup_alpha", type=float, default=0.2)   # set 0 to disable
    ap.add_argument("--label_jitter", type=float, default=0.02) # on scaled targets
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="runs_cylinder_mixup")
    return ap.parse_args()

# ---------- Robust loader (comma OR whitespace) ----------
def load_table(path: str) -> pd.DataFrame:
    """
    Try comma CSV first; if it collapses to 1 column, fall back to whitespace-delimited.
    """
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            raise ValueError("Likely whitespace-delimited; retrying with sep='\\s+'")
        return df
    except Exception:
        return pd.read_csv(path, sep=r"\s+", engine="python")

# ---------- Dataset ----------
class SensorDataset(Dataset):
    def __init__(self, df: pd.DataFrame, fit_norm: bool = False, mean_=None, std_=None):
        # Node columns: prefer names starting with node_ or node, else take first 16 cols
        node_cols = [c for c in df.columns if c.startswith("node_")]
        if len(node_cols) != 16:
            node_cols = [c for c in df.columns if c.startswith("node")]
        if len(node_cols) == 16:
            X = df[node_cols].to_numpy(dtype=np.float32)
        else:
            X = df.iloc[:, :16].to_numpy(dtype=np.float32)

        # Target: accept 'kappa', 'kappa_x', or 'label'
        target_col = None
        for cand in ["kappa", "kappa_x", "label"]:
            if cand in df.columns:
                target_col = cand
                break
        if target_col is None:
            raise ValueError("CSV must have target column: 'kappa' or 'kappa_x' or 'label'.")

        y = df[target_col].to_numpy(dtype=np.float32)

        self.X = X
        self.y = y.reshape(-1, 1)

        # Per-node z-score normalization (fit on train split)
        if fit_norm:
            self.mean_ = self.X.mean(axis=0, keepdims=True)
            self.std_  = self.X.std(axis=0, keepdims=True) + 1e-8
        else:
            self.mean_ = mean_
            self.std_  = std_
        self.X = (self.X - self.mean_) / self.std_

        # Scale targets (1/m) to ~0..0.8 by dividing by 100
        self.y_scaled = self.y / 100.0

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.X[idx]),
                torch.from_numpy(self.y_scaled[idx]))

# ---------- Model ----------
class SmallMLP1D(nn.Module):
    def __init__(self, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 128), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(64, 32),  nn.ReLU(),
            nn.Linear(32, 16),  nn.ReLU(),
            nn.Linear(16, 1)    # output κ (scaled)
        )
    def forward(self, x): return self.net(x)

# ---------- Mixup ----------
def mixup(X, y, alpha=0.3):
    if alpha is None or alpha <= 0: return X, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(X.size(0))
    X_mix = lam * X + (1 - lam) * X[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    return X_mix, y_mix

# ---------- Train/Eval ----------
def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data (comma or whitespace)
    df = load_table(args.csv)

    # Split indices
    N = len(df)
    n_test = int(args.test_ratio * N)
    n_val  = int(args.val_ratio  * N)
    n_train= N - n_val - n_test
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(f"Dataset too small for requested split ratios: N={N}")

    train_df = df.sample(n=n_train, random_state=args.seed)
    remain_df= df.drop(train_df.index)
    val_df   = remain_df.sample(n=n_val, random_state=args.seed+1)
    test_df  = remain_df.drop(val_df.index)

    # Build datasets (fit normalization on train only)
    train_set = SensorDataset(train_df, fit_norm=True)
    val_set   = SensorDataset(val_df,   fit_norm=False, mean_=train_set.mean_, std_=train_set.std_)
    test_set  = SensorDataset(test_df,  fit_norm=False, mean_=train_set.mean_, std_=train_set.std_)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False)

    # Model / loss / optim
    model = SmallMLP1D(dropout=args.dropout)
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_path = os.path.join(args.outdir, "best.pt")

    # Train
    for ep in range(1, args.epochs + 1):
        model.train(); train_loss = 0.0
        for Xb, yb in train_loader:
            # Mixup + label jitter (labels are scaled)
            Xb, yb = mixup(Xb, yb, alpha=args.mixup_alpha)
            if args.label_jitter and args.label_jitter > 0:
                yb = yb + args.label_jitter * torch.randn_like(yb)

            opt.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward(); opt.step()
            train_loss += loss.item()

        # Validation (no mixup)
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                pred = model(Xb)
                val_loss += criterion(pred, yb).item()

        train_loss /= max(1, len(train_loader))
        val_loss   /= max(1, len(val_loader))

        # Save best model; save mean/std as TENSORS to avoid pickle issues
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "mean": torch.from_numpy(train_set.mean_).float(),
                "std":  torch.from_numpy(train_set.std_).float(),
            }, best_path)

        if ep % 10 == 0 or ep == 1:
            print(f"Epoch {ep:03d} | train {train_loss:.5f} | val {val_loss:.5f}")

    # Load best and evaluate on test (report in original 1/m units)
    # PyTorch 2.6 default weights_only=True works because we saved only tensors
    ckpt = torch.load(best_path, map_location="cpu")  # weights_only=True by default
    model.load_state_dict(ckpt["model"])
    mean_ = ckpt["mean"].numpy()
    std_  = ckpt["std"].numpy()

    # (mean_/std_ are not used in evaluation here because the Dataset already applied them,
    #  but they are available if you need to normalize live data later.)

    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for Xb, yb in test_loader:
            p = model(Xb).cpu().numpy() * 100.0   # unscale back to 1/m
            t = yb.cpu().numpy() * 100.0
            preds.append(p); trues.append(t)
    preds = np.vstack(preds).ravel()
    trues = np.vstack(trues).ravel()

    mse  = mean_squared_error(trues, preds)
    mae  = mean_absolute_error(trues, preds)
    r2   = r2_score(trues, preds)
    print(f"\nTest metrics (1/m): RMSE={mse:.3f}, MAE={mae:.3f}, R2={r2:.4f}")

    # Scatter plot: Pred vs True
    plt.figure(figsize=(5,5))
    plt.scatter(trues, preds, s=16)
    lo, hi = min(trues.min(), preds.min()), max(trues.max(), preds.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("True κ (1/m)")
    plt.ylabel("Predicted κ (1/m)")
    plt.title("Cylinder Curvature: Pred vs True")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
