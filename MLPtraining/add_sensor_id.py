# add_sensor_id_space.py — keep space-delimited output
import pandas as pd
import numpy as np

INPUT_FILE  = "sensor_data3.csv"             # your original file
OUTPUT_FILE = "sensor_data_with_ids.csv"    # space-delimited output

ALLOWED_KAPPA = {0,10,20,30,40,50,60,70,80}

def read_csv_flex(path: str) -> pd.DataFrame:
    """Read CSV or whitespace-delimited files robustly (no header case changes)."""
    # 1) Try normal CSV
    try:
        df = pd.read_csv(path, engine="python")
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    # 2) Try flexible separator (commas/tabs/spaces)
    try:
        df = pd.read_csv(path, sep=r"[,\t ]+", engine="python", skip_blank_lines=True)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    # 3) Last resort: whitespace
    return pd.read_csv(path, delim_whitespace=True, engine="python")

def trim_header_spaces(df: pd.DataFrame) -> pd.DataFrame:
    """Trim leading/trailing whitespace from column names (preserve original casing)."""
    df = df.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df

def coerce_kappa_series(s: pd.Series) -> pd.Series:
    """Strip spaces and non-numeric junk; coerce to integer-like."""
    s = s.astype(str).str.strip()
    s = s.str.replace(r"[^0-9\.\-]", "", regex=True)  # keep digits, dot, minus
    s = pd.to_numeric(s, errors="coerce")
    s = s.round().astype("Int64")  # allow NA but int-like
    return s

def find_kappa_column(df: pd.DataFrame) -> str:
    """Find the kappa column, preferring exact name; fall back to heuristic."""
    if "kappa" in df.columns:
        return "kappa"
    # try case-insensitive exact
    for c in df.columns:
        if isinstance(c, str) and c.strip().lower() == "kappa":
            return c
    # heuristic: pick the column whose cleaned values most resemble allowed kappas
    best_col, best_score = None, -1.0
    for c in df.columns:
        ser = coerce_kappa_series(df[c])
        non_na = ser.dropna()
        if len(non_na) == 0:
            continue
        frac_allowed = (non_na.isin(list(ALLOWED_KAPPA))).mean()
        if frac_allowed > best_score:
            best_score, best_col = frac_allowed, c
    if best_col is not None and best_score >= 0.6:
        return best_col
    raise ValueError("Could not locate a 'kappa' column.")

def main():
    df = read_csv_flex(INPUT_FILE)
    if df.empty:
        raise ValueError("Input appears empty or could not be parsed.")

    # Preserve header casing; just trim spaces
    df = trim_header_spaces(df)

    # Locate and clean kappa
    kcol = find_kappa_column(df)
    df[kcol] = coerce_kappa_series(df[kcol])

    # Drop rows where kappa couldn't be parsed
    before = len(df)
    df = df.dropna(subset=[kcol]).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("All rows had invalid 'kappa' after cleaning.")
    df[kcol] = df[kcol].astype(int)

    # Build sensor_id by detecting 80 -> 0 reset in acquisition order
    kappas = df[kcol].to_numpy()
    sensor_ids = np.zeros(len(df), dtype=int)
    sid = 1
    prev = None
    for i, k in enumerate(kappas):
        if prev is not None and prev == 80 and k == 0:
            sid += 1
        sensor_ids[i] = sid
        prev = k

    # Insert sensor_id right after kappa (preserve other column order)
    insert_at = df.columns.get_loc(kcol) + 1
    df.insert(insert_at, "sensor_id", sensor_ids)

    # Save with spaces (single space delimiter)
    # Note: this will normalize multiple spaces to single spaces between columns.
    df.to_csv(OUTPUT_FILE, sep=" ", index=False)

    # Summary
    print(f"Loaded rows: {before}  |  Kept rows: {len(df)}")
    print(f"Detected kappa column: '{kcol}'")
    print(f"Sensors detected: {sensor_ids.max()}")
    print("Rows per sensor_id:")
    print(df.groupby("sensor_id")[kcol].count().to_string())
    print(f"\nSaved (space-delimited): {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
