import matplotlib.pyplot as plt
from nilearn import plotting, image

mri_path = "data/AD/014_S_10351/brain.nii"
atlas_path = "atlas/AAL.nii"

mri = image.load_img(mri_path)
atlas = image.load_img(atlas_path)

# Resampling is for visualization only; it does not perform registration.
atlas_resampled = image.resample_to_img(
    atlas,
    mri,
    interpolation="nearest"
)

display = plotting.plot_anat(
    mri,
    title="MRI with AAL atlas overlay",
    display_mode="ortho",
    cut_coords=(0, -20, 20),
    draw_cross=False
)

display.add_overlay(
    atlas_resampled,
    cmap="Paired",
    alpha=0.35
)

plt.show()
