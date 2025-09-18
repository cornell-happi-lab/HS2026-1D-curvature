# train_blended_paths.py
import os, argparse
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

NODE_COLS = [f"node{i}" for i in range(1,17)]
EPS = 1e-6

# ---------- Args ----------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="sensor_data_with_ids.csv",
                    help="Data with node1..node16, kappa, optional sensor_id.")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--mixup_alpha", type=float, default=0.0)
    ap.add_argument("--label_jitter", type=float, default=0.0)  # safer while tuning
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="runs_blended")
    # Blending controls
    ap.add_argument("--mix_raw_delta", type=float, default=1.0,
                    help="α in z_blend = (1-α)*z_raw + α*delta_z (0=only raw, 1=only delta)")
    ap.add_argument("--mix_eng", type=float, default=1.0,
                    help="β in eng_blend = (1-β)*eng_raw + β*eng_delta")
    ap.add_argument("--concat", type=int, default=0,
                    help="0=blend to 24 dims (default), 1=concat to 48 dims")
    return ap.parse_args()

# ---------- Robust loader ----------
def load_table(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df.shape[1] < 3: raise ValueError
        return df
    except Exception:
        return pd.read_csv(path, sep=r"[,\t ]+", engine="python")

# ---------- Engineered feats ----------
def engineered_features_from_z(z: np.ndarray) -> np.ndarray:
    sum_z   = z.sum(1, keepdims=True)
    mean_z  = z.mean(1, keepdims=True)
    std_z   = z.std(1, keepdims=True)
    min_z   = z.min(1, keepdims=True)
    max_z   = z.max(1, keepdims=True)
    range_z = max_z - min_z
    l2norm  = np.linalg.norm(z, axis=1, keepdims=True)
    q75     = np.quantile(z, 0.75, axis=1, keepdims=True)
    q25     = np.quantile(z, 0.25, axis=1, keepdims=True)
    iqr     = (q75 - q25)
    feats = np.concatenate([sum_z, mean_z, std_z, min_z, max_z, range_z, l2norm, iqr], axis=1)
    return feats.astype(np.float32)  # (B,8)

# ---------- Per-sensor baseline VECTOR (κ=0 mean per node) ----------
def compute_sensor_baseline_vectors(df: pd.DataFrame):
    if "sensor_id" not in df.columns:
        df = df.copy(); df["sensor_id"] = 1
    bvec_map = {}
    for sid, g in df.groupby("sensor_id"):
        flat = g[g["kappa"] == 0]
        if len(flat) > 0:
            bvec = flat[NODE_COLS].mean(axis=0).to_numpy(dtype=np.float32)   # (16,)
        else:
            bvec = g[NODE_COLS].mean(axis=0).to_numpy(dtype=np.float32)
        bvec_map[int(sid)] = bvec
    return bvec_map

# ---------- Dataset (dual paths + blend) ----------
class SensorDataset(Dataset):
    def __init__(self, df, bvec_map, mix_alpha=0.5, mix_beta=0.5, concat=False,
                 # raw path stats
                 node_mean=None, node_std=None,
                 eng_raw_mean=None, eng_raw_std=None,
                 # delta path stats
                 delta_mean=None, delta_std=None,
                 eng_delta_mean=None, eng_delta_std=None,
                 fit_stats=False):
        X = df[NODE_COLS].to_numpy(dtype=np.float32)            # (B,16)
        y = df["kappa"].to_numpy(dtype=np.float32).reshape(-1,1)
        sids = (df["sensor_id"].to_numpy()
                if "sensor_id" in df.columns else np.ones(len(df), dtype=int))
        # Path A: raw z
        if fit_stats:
            node_mean = X.mean(axis=0, keepdims=True)
            node_std  = np.clip(X.std(axis=0, keepdims=True), EPS, None)
        z_raw = (X - node_mean) / node_std                      # (B,16)
        eng_raw = engineered_features_from_z(z_raw)
        if fit_stats:
            eng_raw_mean = eng_raw.mean(axis=0, keepdims=True)
            eng_raw_std  = np.clip(eng_raw.std(axis=0, keepdims=True), EPS, None)
        eng_raw_z = (eng_raw - eng_raw_mean) / eng_raw_std

        # Path B: delta z
        bvec   = np.stack([bvec_map[int(s)] for s in sids], axis=0)   # (B,16)
        delta  = X - bvec
        if fit_stats:
            delta_mean = delta.mean(axis=0, keepdims=True)
            delta_std  = np.clip(delta.std(axis=0, keepdims=True), EPS, None)
        delta_z = (delta - delta_mean) / delta_std
        eng_delta = engineered_features_from_z(delta_z)
        if fit_stats:
            eng_delta_mean = eng_delta.mean(axis=0, keepdims=True)
            eng_delta_std  = np.clip(eng_delta.std(axis=0, keepdims=True), EPS, None)
        eng_delta_z = (eng_delta - eng_delta_mean) / eng_delta_std

        # Blend or concat
        if concat:
            X_cat = np.concatenate([z_raw, delta_z, eng_raw_z, eng_delta_z], axis=1)  # 48
        else:
            alpha = float(mix_alpha); beta = float(mix_beta)
            z_blend   = (1.0 - alpha) * z_raw   + alpha * delta_z         # 16
            eng_blend = (1.0 - beta)  * eng_raw_z + beta  * eng_delta_z   # 8
            X_cat = np.concatenate([z_blend, eng_blend], axis=1)          # 24

        self.X = X_cat.astype(np.float32)
        self.y = (y / 100.0).astype(np.float32)

        # expose stats for inference
        self.node_mean = node_mean; self.node_std = node_std
        self.eng_raw_mean = eng_raw_mean; self.eng_raw_std = eng_raw_std
        self.delta_mean = delta_mean; self.delta_std = delta_std
        self.eng_delta_mean = eng_delta_mean; self.eng_delta_std = eng_delta_std
        self.mix_alpha = mix_alpha; self.mix_beta = mix_beta; self.concat = bool(concat)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])

# ---------- Model ----------
class ResBlock(nn.Module):
    def __init__(self, d, p=0.10):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Dropout(p=p), nn.Linear(d, d))
        self.act = nn.SiLU()
    def forward(self, x): return self.act(x + self.seq(x))

class StrongMLP(nn.Module):
    def __init__(self, in_dim=24, dropout=0.05):
        super().__init__()
        w = 128
        self.stem = nn.Sequential(nn.Linear(in_dim, w), nn.LayerNorm(w), nn.SiLU(), nn.Dropout(p=dropout))
        self.blocks = nn.Sequential(ResBlock(w, dropout), ResBlock(w, dropout), ResBlock(w, dropout))
        self.head = nn.Sequential(nn.Linear(w, 64), nn.SiLU(), nn.Dropout(p=dropout), nn.Linear(64, 1))
    def forward(self, x): x = self.stem(x); x = self.blocks(x); return self.head(x)

# ---------- Mixup ----------
def mixup(X, y, alpha=0.0):
    if alpha is None or alpha <= 0: return X, y
    lam = np.random.beta(alpha, alpha); idx = torch.randperm(X.size(0))
    return lam * X + (1 - lam) * X[idx], lam * y + (1 - lam) * y[idx]

# ---------- Splitting ----------
def smart_split(df, val_ratio, test_ratio, seed):
    N = len(df)
    if "sensor_id" in df.columns:
        g1 = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        tv_idx, te_idx = next(g1.split(np.zeros(N), groups=df["sensor_id"].values))
        tv = df.iloc[tv_idx].reset_index(drop=True)
        te = df.iloc[te_idx].reset_index(drop=True)
        g2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio/(1-test_ratio), random_state=seed+1)
        tr_idx, va_idx = next(g2.split(np.zeros(len(tv)), groups=tv["sensor_id"].values))
        tr = tv.iloc[tr_idx].reset_index(drop=True)
        va = tv.iloc[va_idx].reset_index(drop=True)
    else:
        y = df["kappa"].astype(int).to_numpy()
        n_test = int(round(test_ratio * N))
        n_val  = int(round(val_ratio  * N))
        from sklearn.model_selection import StratifiedShuffleSplit
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=seed)
        tv_idx, te_idx = next(sss1.split(np.zeros(N), y))
        tv = df.iloc[tv_idx].reset_index(drop=True)
        te = df.iloc[te_idx].reset_index(drop=True)
        y_tv = tv["kappa"].astype(int).to_numpy()
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=n_val, random_state=seed+1)
        tr_idx, va_idx = next(sss2.split(np.zeros(len(tv)), y_tv))
        tr = tv.iloc[tr_idx].reset_index(drop=True)
        va = tv.iloc[va_idx].reset_index(drop=True)
    return tr, va, te

# ---------- Train/Eval ----------
def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    df = load_table(args.csv)
    for c in NODE_COLS + ["kappa"]:
        if c not in df.columns: raise ValueError(f"Missing column '{c}'")
    for c in NODE_COLS: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["kappa"] = pd.to_numeric(df["kappa"], errors="coerce").astype(int)
    if "sensor_id" in df.columns:
        df["sensor_id"] = (pd.to_numeric(df["sensor_id"], errors="coerce")
                             .fillna(method="ffill").fillna(method="bfill").astype(int))
    else:
        df["sensor_id"] = 1
    df = df.dropna(subset=NODE_COLS + ["kappa"]).reset_index(drop=True)

    bvec_map = compute_sensor_baseline_vectors(df)
    train_df, val_df, test_df = smart_split(df, args.val_ratio, args.test_ratio, args.seed)

    # Datasets
    train_set = SensorDataset(train_df, bvec_map, args.mix_raw_delta, args.mix_eng, args.concat, fit_stats=True)
    val_set   = SensorDataset(val_df,  bvec_map, args.mix_raw_delta, args.mix_eng, args.concat,
                              node_mean=train_set.node_mean, node_std=train_set.node_std,
                              eng_raw_mean=train_set.eng_raw_mean, eng_raw_std=train_set.eng_raw_std,
                              delta_mean=train_set.delta_mean, delta_std=train_set.delta_std,
                              eng_delta_mean=train_set.eng_delta_mean, eng_delta_std=train_set.eng_delta_std)
    test_set  = SensorDataset(test_df, bvec_map, args.mix_raw_delta, args.mix_eng, args.concat,
                              node_mean=train_set.node_mean, node_std=train_set.node_std,
                              eng_raw_mean=train_set.eng_raw_mean, eng_raw_std=train_set.eng_raw_std,
                              delta_mean=train_set.delta_mean, delta_std=train_set.delta_std,
                              eng_delta_mean=train_set.eng_delta_mean, eng_delta_std=train_set.eng_delta_std)

    in_dim = train_set.X.shape[1]  # 24 (blend) or 48 (concat)
    trL = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    vaL = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False)
    teL = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False)

    model = StrongMLP(in_dim=in_dim, dropout=args.dropout)
    criterion = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    best_val = float("inf"); bad=0; patience=999
    best_path = os.path.join(args.outdir, "best.pt")

    for ep in range(1, args.epochs+1):
        model.train(); tr_loss=0.0
        for xb, yb in trL:
            xb, yb = mixup(xb, yb, alpha=args.mixup_alpha)
            if args.label_jitter>0: yb = yb + args.label_jitter*torch.randn_like(yb)
            opt.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); opt.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(trL))

        model.eval(); va_loss=0.0
        with torch.no_grad():
            for xb, yb in vaL:
                va_loss += criterion(model(xb), yb).item()
        va_loss /= max(1, len(vaL)); scheduler.step()

        if va_loss < best_val:
            best_val, bad = va_loss, 0
            torch.save({
                "model": model.state_dict(),
                "in_dim": in_dim,
                # raw path stats
                "node_mean": torch.from_numpy(train_set.node_mean).float(),
                "node_std":  torch.from_numpy(train_set.node_std).float(),
                "eng_raw_mean": torch.from_numpy(train_set.eng_raw_mean).float(),
                "eng_raw_std":  torch.from_numpy(train_set.eng_raw_std).float(),
                # delta path stats
                "delta_mean": torch.from_numpy(train_set.delta_mean).float(),
                "delta_std":  torch.from_numpy(train_set.delta_std).float(),
                "eng_delta_mean": torch.from_numpy(train_set.eng_delta_mean).float(),
                "eng_delta_std":  torch.from_numpy(train_set.eng_delta_std).float(),
                # blend metadata
                "mix_raw_delta": float(args.mix_raw_delta),
                "mix_eng": float(args.mix_eng),
                "concat": int(args.concat),
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

    plt.figure(figsize=(5,5))
    plt.scatter(trues, preds, s=16)
    lo, hi = min(trues.min(), preds.min()), max(trues.max(), preds.max())
    plt.plot([lo,hi],[lo,hi],"--")
    title = "Pred vs True (blend)" if not args.concat else "Pred vs True (concat)"
    plt.title(title + f" α={args.mix_raw_delta}, β={args.mix_eng}")
    plt.xlabel("True κ (1/m)"); plt.ylabel("Pred κ (1/m)")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
