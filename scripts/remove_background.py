from pathlib import Path
import pandas as pd

root = Path("features")
fixed = 0

for csv_file in root.glob("*/*.csv"):
    df = pd.read_csv(csv_file)

    if "ROI_ID" in df.columns:
        cleaned = df[df["ROI_ID"] != 0].copy()

        if len(cleaned) == 116:
            cleaned.to_csv(csv_file, index=False)
            fixed += 1
        else:
            print(f"[CHECK] {csv_file}: {len(cleaned)} non-background rows")

print(f"Fixed CSV files: {fixed}")
