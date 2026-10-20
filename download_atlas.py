from nilearn import datasets

aal = datasets.fetch_atlas_aal(version='SPM12')

print(aal.maps)
print(aal.labels)