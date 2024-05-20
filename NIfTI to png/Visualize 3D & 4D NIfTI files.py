import os
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting

# Define input and output directories
input_dir = '/home/orion/Geo/UCLA data/FMRIPrep/visualization-test/'
output_dir = '/home/orion/Geo/UCLA data/FMRIPrep/visualization-output/'

# Create output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Search for brain mask and preprocessed fMRI files
fmri_files = [f for f in os.listdir(input_dir) if f.endswith('_preproc.nii.gz')]
brainmask_files = [f for f in os.listdir(input_dir) if f.endswith('_brainmask.nii.gz')]

# Load, visualize, and save the images
for fmri_file, brainmask_file in zip(fmri_files, brainmask_files):
    fmri_img = nib.load(os.path.join(input_dir, fmri_file))
    brainmask_img = nib.load(os.path.join(input_dir, brainmask_file))

    # Get the number of time points in the 4D fMRI data
    num_time_points = fmri_img.shape[-1]
    print(f"File: {fmri_file} - Number of time points: {num_time_points}")

    # Plot and save the 4D fMRI data (showing one time point as an example)
    time_point = 42  # Change this to visualize a different time point
    display = plotting.plot_stat_map(nib.Nifti1Image(fmri_img.get_fdata()[..., time_point], fmri_img.affine),
                                     title=f"fMRI Time Point {time_point}")
    output_fmri_file = os.path.join(output_dir, f"{fmri_file}_timepoint{time_point}.png")
    display.savefig(output_fmri_file)
    display.close()

    # Plot and save the 3D brain mask
    display = plotting.plot_roi(brainmask_img, title="Brain Mask")
    output_brainmask_file = os.path.join(output_dir, f"{brainmask_file}.png")
    display.savefig(output_brainmask_file)
    display.close()

    print(f"Saved visualizations for {fmri_file} and {brainmask_file} to {output_dir}")

# If there are files that do not have a corresponding pair
if len(fmri_files) != len(brainmask_files):
    print("Warning: The number of fMRI and brain mask files do not match.")
    print(f"Found {len(fmri_files)} fMRI files and {len(brainmask_files)} brain mask files.")
