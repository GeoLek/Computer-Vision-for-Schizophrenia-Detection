import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

# Define your data directories
data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/Applied Brain Mask & Regressed out confounds/'
new_data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/4D_Downsized/'  # New directory to save downsized images

# Create new data directory if it doesn't exist
os.makedirs(new_data_dir, exist_ok=True)


# Function to downsize the 3D dimensions
def downsize_image(image, zoom_factor=(0.5, 0.5, 0.5)):
    return zoom(image, zoom_factor, order=3)  # Using cubic interpolation


# Iterate through groups
for group in ['SCHZ', 'HC']:
    group_dir = os.path.join(data_dir, group)
    new_group_dir = os.path.join(new_data_dir, group)
    os.makedirs(new_group_dir, exist_ok=True)

    label = 0 if group == 'SCHZ' else 1

    # Iterate through subjects
    for subject in os.listdir(group_dir):
        subject_dir = os.path.join(group_dir, subject, 'func')
        new_subject_dir = os.path.join(new_group_dir, subject, 'func')
        os.makedirs(new_subject_dir, exist_ok=True)

        # Iterate through task files
        for task_file in os.listdir(subject_dir):
            if 'preproc.nii.gz' in task_file:
                bold_path = os.path.join(subject_dir, task_file)
                new_bold_path = os.path.join(new_subject_dir, task_file)

                # Load the NIfTI file
                img = nib.load(bold_path)
                data = img.get_fdata()

                # Downsize each 3D volume while keeping the 4th dimension
                downsized_data = np.array([downsize_image(data[..., t]) for t in range(data.shape[-1])])
                downsized_data = np.moveaxis(downsized_data, 0, -1)  # Correct axis order

                # Save the downsized image
                downsized_img = nib.Nifti1Image(downsized_data, img.affine, img.header)
                nib.save(downsized_img, new_bold_path)

print("Downsizing complete. New images saved to:", new_data_dir)
