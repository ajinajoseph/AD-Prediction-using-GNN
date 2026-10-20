"""Extract five intensity features for each non-background AAL ROI.
Run from project root: python scripts/extract_features.py
Input: data/<AD|MCI|CN>/<subject>/brain.nii[.gz]
Atlas: atlas/AAL.nii[.gz], atlas/aal_labels.csv (Index,Region)
"""
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["AD", "MCI", "CN"]
FEATURES = ["Mean", "Std", "Median", "Max", "Min"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", choices=CLASSES, default=CLASSES)
    args = parser.parse_args()
    atlas_path = next((p for p in [ROOT/"atlas/AAL.nii", ROOT/"atlas/AAL.nii.gz"] if p.exists()), None)
    labels_path = next((p for p in [ROOT/"atlas/aal_labels.csv", ROOT/"atlas/AAL_labels.csv"] if p.exists()), None)
    if atlas_path is None:
        raise FileNotFoundError("Put AAL.nii or AAL.nii.gz in atlas/")
    if labels_path is None:
        raise FileNotFoundError("Put aal_labels.csv in atlas/ with columns Index,Region")
    labels = pd.read_csv(labels_path)
    if not {"Index","Region"}.issubset(labels.columns):
        raise ValueError("Label CSV must have Index and Region columns")
    labels["Index"] = pd.to_numeric(labels["Index"], errors="raise").astype(int)
    labels = labels[labels["Index"] != 0].drop_duplicates("Index").sort_values("Index")
    print(f"Atlas labels: {len(labels)} non-background ROIs (expected 116)")
    for cls in args.classes:
        in_dir, out_dir = ROOT/"data"/cls, ROOT/"features"/cls
        out_dir.mkdir(parents=True, exist_ok=True)
        subjects = sorted(p for p in in_dir.iterdir() if p.is_dir()) if in_dir.exists() else []
        print(f"\\n{cls}: found {len(subjects)} subject folders")
        for subject in subjects:
            brain = next((p for p in [subject/"brain.nii", subject/"brain.nii.gz"] if p.exists()), None)
            if brain is None:
                print(f"SKIP {subject.name}: brain.nii/.nii.gz not found")
                continue
            try:
                img = nib.load(str(brain))
                atlas = nib.load(str(atlas_path))
                # Resampling matches grids only; atlas must already be anatomically aligned to MRI.
                atlas_img = resample_to_img(atlas, img, interpolation="nearest")
                image_data = img.get_fdata(dtype=np.float32)
                atlas_data = np.rint(atlas_img.get_fdata(dtype=np.float32)).astype(np.int32)
                rows, missing = [], []
                for _, row in labels.iterrows():
                    roi_id, name = int(row["Index"]), str(row["Region"])
                    voxels = image_data[atlas_data == roi_id]
                    voxels = voxels[np.isfinite(voxels)]
                    if voxels.size == 0:
                        missing.append(name)
                        stats = [0.0]*5
                    else:
                        stats = [float(np.mean(voxels)), float(np.std(voxels)),
                                 float(np.median(voxels)), float(np.max(voxels)),
                                 float(np.min(voxels))]
                    rows.append([roi_id, name, *stats])
                df = pd.DataFrame(rows, columns=["ROI_ID","ROI_Name",*FEATURES])
                df.to_csv(out_dir/f"{subject.name}.csv", index=False)
                print(f"Saved {cls}/{subject.name}.csv ({len(df)} ROIs)")
                if missing:
                    print(f"  WARNING: {len(missing)} ROIs had no voxels; first few: {missing[:5]}")
            except Exception as e:
                print(f"ERROR {subject.name}: {e}")
if __name__ == "__main__":
    main()
