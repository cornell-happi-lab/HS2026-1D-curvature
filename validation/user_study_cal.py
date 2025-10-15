import pandas as pd

def analyze_file(path):
    # Load CSV file
    df = pd.read_csv(path)

    # Only keep the columns we need
    cols = ["force_c0", "force_curv", "ft_z"]
    if not all(c in df.columns for c in cols):
        raise ValueError(f"File is missing one of the required columns: {cols}")

    # Compute mean and std
    stats = df[cols].agg(["mean", "std"]).T
    stats.columns = ["average", "stddev"]

    return stats

if __name__ == "__main__":
    filename = "hold_20251012_135230_refNA_curv25.csv"  # <-- replace with your actual file
    stats = analyze_file(filename)
    print("Summary statistics:\n")
    print(stats)
