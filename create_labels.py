from nilearn import datasets
import pandas as pd

aal = datasets.fetch_atlas_aal(version="SPM12")

df = pd.DataFrame({
    "Index": aal.indices,
    "Region": aal.labels
})

df.to_csv("atlas/aal_labels.csv", index=False)

print(df.head())