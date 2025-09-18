# train_cnn_full.py
import os, argparse
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# -------------------- Config --------------------
NODE_COLS = [f"node{i}" for i in range(1,17)]

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, default="sensor_data_with_ids.csv", help="CSV with node1..node16, kappa, sensor_id")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--mixup_alpha", type=float, default=0.15)
    ap.add_argument("--label_jitter", type=float, default=0.01)  # on scaled targets
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="runs_cnn_full")
    # toggles
    ap.add_argument("--use_feats", type=int, default=1, choices=[0,1], help="use engineered features")
    ap.add_argument("--use_baseline", type=int, default=1, choices=[0,1], help="add baseline-relative channel")
    ap.add_argument("--use_embed", type=int, default=1, choices=[0,1], help="add sensor_id embedding")
    ap.add_argument("--embed_dim", type=int, default=8)
    return ap.parse_args()

# -------------------- IO --------------------
def load_table(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df.shape[1] < 3: raise ValueError
        return df
    except Exception:
        return pd.read_csv(path, sep=r"[,\t ]+", engine="python")

# -------------------- Features --------------------
def engineered_features_from_z(z: np.ndarray) -> np.ndarray:
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

# -------------------- Baselines --------------------
def compute_sensor_baselines(train_df: pd.DataFrame) -> dict:
    """
    Returns dict with string keys:
      { 'sensor_id_str' -> (16,), '__global__' -> (16,) }
    Per-sensor baseline = mean of rows with kappa==0, fallback to that sensor's overall mean.
    Global baseline = mean over kappa==0 in TRAIN, fallback to overall TRAIN mean.
    """
    baselines = {}
    assert "sensor_id" in train_df.columns, "sensor_id column required"

    for sid, g in train_df.groupby("sensor_id"):
        sid = str(sid)  # string keys everywhere
        g0 = g[g["kappa"] == 0]
        if len(g0) > 0:
            base = g0[NODE_COLS].to_numpy(dtype=np.float32).mean(axis=0)
        else:
            base = g[NODE_COLS].to_numpy(dtype=np.float32).mean(axis=0)
        baselines[sid] = base.astype(np.float32)

    all0 = train_df[train_df["kappa"] == 0]
    if len(all0) > 0:
        gbase = all0[NODE_COLS].to_numpy(dtype=np.float32).mean(axis=0)
    else:
        gbase = train_df[NODE_COLS].to_numpy(dtype=np.float32).mean(axis=0)
    baselines["__global__"] = gbase.astype(np.float32)
    return baselines

# -------------------- Dataset --------------------
class SensorDataset(Dataset):
    """
    Yields:
      img  : (C,4,4) where C=1 or 2 (z, and optionally z - baseline_z)
      feats: (8,) engineered features (z-scored)
      sid  : ()  sensor index (long) for embedding
      y    : (1,) scaled target
    """
    def __init__(self, df, node_mean=None, node_std=None,
                 feat_mean=None, feat_std=None,
                 sensor2baseline=None, sid2idx=None, unk_idx=None,
                 fit_stats=False, use_baseline=True):
        self.use_baseline = use_baseline
        self.has_sid = "sensor_id" in df.columns

        # raw
        X_nodes = df[NODE_COLS].to_numpy(dtype=np.float32)
        y = df["kappa"].to_numpy(dtype=np.float32).reshape(-1,1)  # 0..80

        # z-score nodes (fit on train only)
        if fit_stats:
            node_mean = X_nodes.mean(axis=0, keepdims=True)
            node_std  = X_nodes.std(axis=0, keepdims=True) + 1e-8
        z = (X_nodes - node_mean) / node_std

        # engineered features on z
        feats = engineered_features_from_z(z)
        if fit_stats:
            feat_mean = feats.mean(axis=0, keepdims=True)
            feat_std  = feats.std(axis=0, keepdims=True) + 1e-8
        feats_z = (feats - feat_mean) / feat_std

        # baseline channel
        if use_baseline and self.has_sid:
            assert sensor2baseline is not None and sid2idx is not None and unk_idx is not None
            sid_arr = df["sensor_id"].astype(str).to_numpy()
            gbase = sensor2baseline["__global__"]
            base_list = [sensor2baseline.get(s, gbase) for s in sid_arr]
            base = np.stack(base_list, axis=0)  # (B,16)
            base_z = (base - node_mean) / node_std
            delta_z = z - base_z
            img = np.stack([z.reshape(-1,4,4), delta_z.reshape(-1,4,4)], axis=1)  # (B,2,4,4)
        else:
            img = z.reshape(-1,1,4,4)

        self.img  = img.astype(np.float32)
        self.feat = feats_z.astype(np.float32)
        self.y    = (y / 100.0).astype(np.float32)  # 0..0.8

        # sensor indices for embedding (+UNK)
        if self.has_sid:
            self.sid_idx = np.array([sid2idx.get(str(s), unk_idx) for s in df["sensor_id"]], dtype=np.int64)
        else:
            self.sid_idx = np.zeros((len(df),), dtype=np.int64)

        # store stats for reuse
        self.node_mean, self.node_std = node_mean, node_std
        self.feat_mean, self.feat_std = feat_mean, feat_std

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return (torch.from_numpy(self.img[i]),
                torch.from_numpy(self.feat[i]),
                torch.tensor(self.sid_idx[i], dtype=torch.long),
                torch.from_numpy(self.y[i]))

# -------------------- Model --------------------
class FeatMLP(nn.Module):
    def __init__(self, in_dim=8, hid=16, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.SiLU(), nn.Dropout(p=dropout),
        )
    def forward(self, x): return self.net(x)

class SmallCNNPlus(nn.Module):
    def __init__(self, in_ch=1, use_feats=True, use_embed=True, n_sensors=1, embed_dim=8, dropout=0.10):
        super().__init__()
        self.use_feats = use_feats
        self.use_embed = use_embed

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.SiLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.img_drop = nn.Dropout(p=dropout)

        feat_dim = 0
        if use_feats:
            self.feat_mlp = FeatMLP(in_dim=8, hid=16, dropout=dropout)
            feat_dim += 16
        if use_embed:
            self.emb = nn.Embedding(num_embeddings=n_sensors, embedding_dim=embed_dim)
            feat_dim += embed_dim

        fused = 32 + feat_dim
        self.head = nn.Sequential(
            nn.Linear(fused, 64), nn.SiLU(), nn.Dropout(p=dropout),
            nn.Linear(64, 1)
        )

    def forward(self, img, feats=None, sid=None):
        h = self.conv(img)
        h = self.gap(h).flatten(1)  # (B,32)
        h = self.img_drop(h)

        extras = []
        if self.use_feats and feats is not None:
            extras.append(self.feat_mlp(feats))
        if self.use_embed and sid is not None:
            extras.append(self.emb(sid))
        if extras:
            h = torch.cat([h] + extras, dim=1)
        return self.head(h)

# -------------------- Utils --------------------
def mixup(img, feats, y, alpha=0.15):
    if alpha is None or alpha <= 0: return img, feats, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(img.size(0), device=img.device)
    return lam*img + (1-lam)*img[idx], lam*feats + (1-lam)*feats[idx], lam*y + (1-lam)*y[idx]

def smart_split(df, val_ratio, test_ratio, seed):
    N = len(df)
    if "sensor_id" in df.columns:
        gss1 = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        train_val_idx, test_idx = next(gss1.split(np.zeros(N), groups=df["sensor_id"].values))
        tv = df.iloc[train_val_idx].reset_index(drop=True)
        te = df.iloc[test_idx].reset_index(drop=True)

        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio / (1 - test_ratio), random_state=seed+1)
        train_idx, val_idx = next(gss2.split(np.zeros(len(tv)), groups=tv["sensor_id"].values))
        tr = tv.iloc[train_idx].reset_index(drop=True)
        va = tv.iloc[val_idx].reset_index(drop=True)
    else:
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

def evaluate(model, loader, criterion, device, use_feats, use_embed):
    model.eval(); tot = 0.0
    with torch.no_grad():
        for img, feats, sid, y in loader:
            img, feats, sid, y = img.to(device), feats.to(device), sid.to(device), y.to(device)
            pred = model(img,
                         feats if use_feats else None,
                         sid if use_embed else None)
            tot += criterion(pred, y).item()
    return tot / max(1, len(loader))

# -------------------- Main --------------------
def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # Load & clean
    df = load_table(args.csv)
    # Basic checks
    for c in NODE_COLS + ["kappa", "sensor_id"]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {args.csv}")
    # Cast
    for c in NODE_COLS: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["kappa"] = pd.to_numeric(df["kappa"], errors="coerce").astype(int)
    # Standardize sensor_id to string early
    df["sensor_id"] = df["sensor_id"].astype(str)
    df = df.dropna(subset=NODE_COLS + ["kappa", "sensor_id"]).reset_index(drop=True)

    # Split (group-aware by sensor_id)
    train_df, val_df, test_df = smart_split(df, args.val_ratio, args.test_ratio, args.seed)

    # Baselines on TRAIN only + sensor mapping with UNK
    sensor2baseline = compute_sensor_baselines(train_df) if args.use_baseline else None
    uniq_sids = sorted(train_df["sensor_id"].unique().tolist()) if args.use_embed else []
    sid2idx = {s:i for i,s in enumerate(uniq_sids)} if args.use_embed else {}
    unk_idx = len(sid2idx) if args.use_embed else 0
    n_sensors = (len(sid2idx) + 1) if args.use_embed else 1  # +UNK

    # Datasets (fit stats on train only)
    train_set = SensorDataset(train_df, fit_stats=True,
                              sensor2baseline=sensor2baseline, sid2idx=sid2idx, unk_idx=unk_idx,
                              use_baseline=bool(args.use_baseline))
    val_set   = SensorDataset(val_df,
                              node_mean=train_set.node_mean, node_std=train_set.node_std,
                              feat_mean=train_set.feat_mean, feat_std=train_set.feat_std,
                              sensor2baseline=sensor2baseline, sid2idx=sid2idx, unk_idx=unk_idx,
                              use_baseline=bool(args.use_baseline))
    test_set  = SensorDataset(test_df,
                              node_mean=train_set.node_mean, node_std=train_set.node_std,
                              feat_mean=train_set.feat_mean, feat_std=train_set.feat_std,
                              sensor2baseline=sensor2baseline, sid2idx=sid2idx, unk_idx=unk_idx,
                              use_baseline=bool(args.use_baseline))

    trL = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    vaL = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False)
    teL = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = 2 if args.use_baseline else 1
    model = SmallCNNPlus(in_ch=in_ch,
                         use_feats=bool(args.use_feats),
                         use_embed=bool(args.use_embed),
                         n_sensors=n_sensors,
                         embed_dim=args.embed_dim,
                         dropout=args.dropout).to(device)
    criterion = nn.HuberLoss(delta=0.02)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    best_val = float("inf")
    best_path = os.path.join(args.outdir, "best.pt")
    patience = 999; bad = 0

    for ep in range(1, args.epochs+1):
        # Train
        model.train(); tr_loss = 0.0
        for img, feats, sid, y in trL:
            img, feats, sid, y = img.to(device), feats.to(device), sid.to(device), y.to(device)

            img, feats, y = mixup(img, feats, y, alpha=args.mixup_alpha)
            if args.label_jitter > 0:
                y = y + args.label_jitter * torch.randn_like(y)

            opt.zero_grad()
            pred = model(img,
                         feats if args.use_feats else None,
                         sid if args.use_embed else None)
            loss = criterion(pred, y)
            loss.backward(); opt.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(trL))

        # Validate
        va_loss = evaluate(model, vaL, criterion, device, bool(args.use_feats), bool(args.use_embed))
        scheduler.step()

        if va_loss < best_val:
            best_val, bad = va_loss, 0
            torch.save({
                "model": model.state_dict(),
                "node_mean": torch.from_numpy(train_set.node_mean).float(),
                "node_std":  torch.from_numpy(train_set.node_std).float(),
                "feat_mean": torch.from_numpy(train_set.feat_mean).float(),
                "feat_std":  torch.from_numpy(train_set.feat_std).float(),
                "use_feats": bool(args.use_feats),
                "use_baseline": bool(args.use_baseline),
                "use_embed": bool(args.use_embed),
                "embed_dim": int(args.embed_dim),
                "sid2idx": sid2idx,
            }, best_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {ep} (best val {best_val:.6f})")
                break

        if ep % 10 == 0 or ep == 1:
            print(f"Epoch {ep:03d} | train {tr_loss:.5f} | val {va_loss:.5f}")

    # Test
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    preds, trues = [], []
    with torch.no_grad():
        for img, feats, sid, y in teL:
            img, feats, sid, y = img.to(device), feats.to(device), sid.to(device), y.to(device)
            p = model(img,
                      feats if ckpt["use_feats"] else None,
                      sid if ckpt["use_embed"] else None).cpu().numpy() * 100.0
            t = y.cpu().numpy() * 100.0
            preds.append(p); trues.append(t)
    preds = np.vstack(preds).ravel()
    trues = np.vstack(trues).ravel()

    rmse = float(np.sqrt(mean_squared_error(trues, preds)))
    mae  = float(mean_absolute_error(trues, preds))
    r2   = float(r2_score(trues, preds))
    print(f"\nTEST (1/m): RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.4f}")

    # Scatter
    plt.figure(figsize=(5,5))
    plt.scatter(trues, preds, s=16)
    lo, hi = min(trues.min(), preds.min()), max(trues.max(), preds.max())
    plt.plot([lo,hi],[lo,hi],"--")
    plt.xlabel("True κ (1/m)"); plt.ylabel("Pred κ (1/m)")
    plt.title("Pred vs True (CNN)")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
