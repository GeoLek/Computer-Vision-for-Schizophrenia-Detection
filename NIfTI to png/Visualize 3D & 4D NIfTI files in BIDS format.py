import os
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting

# Define input and output directories
input_dir = '/home/orion/Geo/UCLA data/FMRIPrep/data'
output_dir = '/home/orion/Geo/UCLA data/FMRIPrep/visualization-output'

# Create output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to find all preprocessed fMRI and brain mask files
def find_files(root_dir):
    fmri_files = []
    brainmask_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('_preproc.nii.gz'):
                fmri_files.append(os.path.join(root, file))
            elif file.endswith('_brainmask.nii.gz'):
                brainmask_files.append(os.path.join(root, file))
    return fmri_files, brainmask_files

# Find files
fmri_files, brainmask_files = find_files(input_dir)

# Create a dictionary to pair fMRI and brain mask files based on subject and task
paired_files = {}
for fmri_file in fmri_files:
    base_name = os.path.basename(fmri_file).replace('_preproc.nii.gz', '')
    paired_files[base_name] = {'fmri': fmri_file}

for brainmask_file in brainmask_files:
    base_name = os.path.basename(brainmask_file).replace('_brainmask.nii.gz', '')
    if base_name in paired_files:
        paired_files[base_name]['brainmask'] = brainmask_file

# Load, visualize, and save the images
for base_name, files in paired_files.items():
    if 'fmri' in files and 'brainmask' in files:
        fmri_img = nib.load(files['fmri'])
        brainmask_img = nib.load(files['brainmask'])

        # Get the number of time points in the 4D fMRI data
        num_time_points = fmri_img.shape[-1]
        print(f"File: {files['fmri']} - Number of time points: {num_time_points}")

        # Plot and save the 4D fMRI data (showing one time point as an example)
        time_point = 0  # Change this to visualize a different time point
        display = plotting.plot_stat_map(nib.Nifti1Image(fmri_img.get_fdata()[..., time_point], fmri_img.affine), title=f"fMRI Time Point {time_point}")
        output_fmri_file = os.path.join(output_dir, f"{base_name}_timepoint{time_point}.png")
        display.savefig(output_fmri_file)
        display.close()

        # Plot and save the 3D brain mask
        display = plotting.plot_roi(brainmask_img, title="Brain Mask")
        output_brainmask_file = os.path.join(output_dir, f"{base_name}_brainmask.png")
        display.savefig(output_brainmask_file)
        display.close()

        print(f"Saved visualizations for {base_name} to {output_dir}")

# If there are files that do not have a corresponding pair
for base_name, files in paired_files.items():
    if 'fmri' not in files or 'brainmask' not in files:
        print(f"Warning: Missing pair for {base_name}")
