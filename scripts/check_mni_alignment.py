import matplotlib.pyplot as plt
from nilearn import image, plotting
import os

atlas_path = "atlas/AAL.nii"
mni_mri_path = os.path.join(
    os.environ["USERPROFILE"],
    "Downloads", "FreeSurfer_AD", "014_S_10351",
    "mri", "transforms", "synthmorph.1.0mm.1.0mm",
    "test.nii.gz"
)

atlas = image.load_img(atlas_path)
mni_mri = image.load_img(mni_mri_path)

atlas_mni = image.resample_to_img(
    atlas, mni_mri, interpolation="nearest"
)

display = plotting.plot_anat(
    mni_mri,
    title="Subject MRI in MNI152 space",
    display_mode="ortho",
    cut_coords=(0, -20, 20),
    draw_cross=False
)

display.add_overlay(atlas_mni, cmap="Paired", alpha=0.35)
plt.show()
