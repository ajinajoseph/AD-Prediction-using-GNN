from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", default=["AD", "MCI", "CN"])
    args = parser.parse_args()

    atlas_path = ROOT / "atlas" / "AAL.nii"
    labels_path = ROOT / "atlas" / "aal_labels.csv"

    if not atlas_path.exists():
        raise FileNotFoundError(f"AAL atlas not found: {atlas_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"AAL labels CSV not found: {labels_path}")

    atlas_img = nib.load(str(atlas_path))
    labels_df = pd.read_csv(labels_path)

    # Detect the ROI label and name columns in the labels CSV.
    cols = {c.lower().strip(): c for c in labels_df.columns}
    id_col = next((cols[k] for k in
                   ["roi_id", "id", "index", "label", "value"]
                   if k in cols), None)
    name_col = next((cols[k] for k in
                     ["roi_name", "roi", "name", "region", "region_name"]
                     if k in cols), None)

    if id_col is None or name_col is None:
        raise ValueError(
            f"Could not identify ROI ID/name columns in {labels_df.columns.tolist()}"
        )

    out_root = ROOT / "features"
    total = 0
    skipped = []

    for cls in args.classes:
        data_root = ROOT / "data" / cls
        out_dir = out_root / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        if not data_root.exists():
            print(f"Skipping missing class folder: {data_root}")
            continue

        for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
            # Use the MNI-space FreeSurfer output, not native brain.nii.
            image_path = subject_dir / "brain_mni.nii.gz"
            if not image_path.exists():
                skipped.append(f"{cls}/{subject_dir.name}: missing brain_mni.nii.gz")
                continue

            try:
                img = nib.load(str(image_path))
                atlas_resampled = resample_to_img(
                    atlas_img, img, interpolation="nearest",
                    force_resample=True, copy_header=True
                )
                atlas_data = np.rint(atlas_resampled.get_fdata()).astype(np.int32)
                image_data = img.get_fdata(dtype=np.float32)

                rows = []
                for _, label_row in labels_df.iterrows():
                    roi_id = int(label_row[id_col])
                    mask = (atlas_data == roi_id)
                    values = image_data[mask]
                    values = values[np.isfinite(values)]

                    if values.size == 0:
                        rows.append({
                            "ROI_ID": roi_id, "ROI_Name": str(label_row[name_col]),
                            "Mean": 0.0, "Std": 0.0, "Median": 0.0,
                            "Max": 0.0, "Min": 0.0
                        })
                    else:
                        rows.append({
                            "ROI_ID": roi_id, "ROI_Name": str(label_row[name_col]),
                            "Mean": float(np.mean(values)),
                            "Std": float(np.std(values)),
                            "Median": float(np.median(values)),
                            "Max": float(np.max(values)),
                            "Min": float(np.min(values))
                        })

                out_file = out_dir / f"{subject_dir.name}.csv"
                pd.DataFrame(rows).to_csv(out_file, index=False)
                total += 1
                print(f"[OK] {cls}/{subject_dir.name}")

            except Exception as exc:
                skipped.append(f"{cls}/{subject_dir.name}: {exc}")
                print(f"[ERROR] {cls}/{subject_dir.name}: {exc}")

    print(f"\nFeature CSVs generated: {total}")
    print(f"Skipped/failed: {len(skipped)}")
    for item in skipped:
        print(" -", item)

if __name__ == "__main__":
    main()
