# realtime_strong.py
import argparse, sys, time
import numpy as np
import torch, torch.nn as nn
import serial

NODE_COLS = [f"node{i}" for i in range(1,17)]

class ResBlock(nn.Module):
    def __init__(self, d, p=0.10):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(d, d), nn.SiLU(), nn.Dropout(p=p),
            nn.Linear(d, d)
        )
        self.act = nn.SiLU()
    def forward(self, x): return self.act(x + self.seq(x))

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
            nn.Linear(width,64), nn.SiLU(), nn.Dropout(p=dropout),
            nn.Linear(64,1)
        )
    def forward(self, x):
        x = self.stem(x); x = self.blocks(x); return self.head(x)

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
    return np.concatenate([sum_z, mean_z, std_z, min_z, max_z, range_z, l2norm, iqr], axis=1).astype(np.float32)

def load_checkpoint(ckpt_path, dropout=0.10):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    in_dim = int(ckpt.get("in_dim", 24))
    model = StrongMLP(in_dim=in_dim, dropout=dropout)
    model.load_state_dict(ckpt["model"])
    model.eval()

    node_mean = ckpt["node_mean"].numpy()
    node_std  = ckpt["node_std"].numpy()
    feat_mean = ckpt["feat_mean"].numpy()
    feat_std  = ckpt["feat_std"].numpy()
    return model, node_mean, node_std, feat_mean, feat_std, in_dim

def parse_arduino_line(line):
    line = line.strip()
    if not line: raise ValueError("empty")
    parts = [p for p in line.split(",") if p.strip() != ""]
    if len(parts) == 1: parts = line.split()
    if len(parts) != 16: raise ValueError(f"Expected 16 values, got {len(parts)}")
    return np.array([float(p) for p in parts], dtype=np.float32)

def build_input(vals16, node_mean, node_std, feat_mean, feat_std):
    x = vals16.reshape(1,16)
    z = (x - node_mean) / (node_std + 1e-8)        # (1,16)
    feats = engineered_features_from_z(z)          # (1,8)
    feats_z = (feats - feat_mean) / (feat_std + 1e-8)
    X = np.concatenate([z, feats_z], axis=1).astype(np.float32)  # (1,24)
    return torch.from_numpy(X)

def main():
    ap = argparse.ArgumentParser(description="Realtime curvature prediction (raw → z-score → engineered features).")
    ap.add_argument("--port", type=str, required=True, help="COM5 or /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--ckpt", type=str, default="runs_strong/best.pt")
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--smooth", type=int, default=5, help="EMA window (N). Set 0 to disable.")
    ap.add_argument("--alpha", type=float, default=0.4, help="EMA factor if smooth>0 (0..1)")
    ap.add_argument("--max_hz", type=float, default=0.0)
    ap.add_argument("--flush_lines", type=int, default=5)
    args = ap.parse_args()

    model, node_mean, node_std, feat_mean, feat_std, in_dim = load_checkpoint(args.ckpt, args.dropout)

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except Exception as e:
        print(f"ERROR opening serial {args.port}: {e}", file=sys.stderr); sys.exit(1)

    for _ in range(max(0, args.flush_lines)):
        try: ser.readline()
        except Exception: pass

    ema = None
    dt = 0.0 if args.max_hz <= 0 else 1.0/args.max_hz
    last = 0.0

    print("Listening… (Ctrl+C to stop)", file=sys.stderr)
    try:
        while True:
            raw = ser.readline().decode(errors="ignore")
            if not raw: continue
            try:
                vals16 = parse_arduino_line(raw)
                X = build_input(vals16, node_mean, node_std, feat_mean, feat_std)
                with torch.no_grad():
                    pred_scaled = model(X).item()  # ~0..0.8
                kappa = pred_scaled * 100.0

                if args.smooth and args.smooth > 0:
                    # simple EMA: y_t = a*y_{t-1} + (1-a)*x_t ; alpha≈0.3~0.5 works well
                    if ema is None: ema = kappa
                    else: ema = args.alpha*ema + (1-args.alpha)*kappa
                    out = ema
                else:
                    out = kappa

                now = time.time()
                if dt > 0 and (now - last) < dt:
                    time.sleep(dt - (now - last))
                last = time.time()

                print(f"{out:.4f}")  # numeric only
                sys.stdout.flush()

            except Exception as e:
                print(f"PARSE ERROR: {e} | line: {raw.strip()}", file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        try: ser.close()
        except Exception: pass

if __name__ == "__main__":
    main()
