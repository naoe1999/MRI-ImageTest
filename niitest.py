import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def show_slices(slices):
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")


# img_nii = nib.load('img/hippocampus_001.nii')
img_nii = nib.load('img/BRATS_001.nii.gz')
img = np.array(img_nii.get_fdata())

print(type(img))
print(img.shape)

# slice0 = img[17, :, :]
# slice1 = img[:, 25, :]
# slice2 = img[:, :, 17]
slice0 = img[120, :, :, 3]
slice1 = img[:, 120, :, 3]
slice2 = img[:, :, 77, 3]

show_slices([slice0, slice1, slice2])

plt.show()

