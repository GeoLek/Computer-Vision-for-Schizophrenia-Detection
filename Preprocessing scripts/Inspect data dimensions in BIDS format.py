import os
import nibabel as nib


def load_nifti_file(filepath):
    img = nib.load(filepath)
    data = img.get_fdata()
    return data


def inspect_data_dimensions(data_dir):
    dimensions = []

    for group in ['SCHZ', 'HC']:
        group_dir = os.path.join(data_dir, group)

        for subject in os.listdir(group_dir):
            subject_dir = os.path.join(group_dir, subject, 'func')
            for task_file in os.listdir(subject_dir):
                if 'preproc.nii.gz' in task_file:
                    bold_path = os.path.join(subject_dir, task_file)

                    # Load data
                    bold_data = load_nifti_file(bold_path)

                    # Get dimensions
                    dim = bold_data.shape
                    dimensions.append(dim)
                    print(f"File: {bold_path}, Dimensions: {dim}")

    return dimensions


# Define the data directory
data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/data/'

# Inspect data dimensions
dimensions = inspect_data_dimensions(data_dir)

# Print summary statistics about dimensions
import numpy as np

dimensions_array = np.array(dimensions)
print(f"\nSummary of Data Dimensions:")
print(f"Min dimensions: {np.min(dimensions_array, axis=0)}")
print(f"Max dimensions: {np.max(dimensions_array, axis=0)}")
print(f"Mean dimensions: {np.mean(dimensions_array, axis=0)}")
print(f"Standard deviation of dimensions: {np.std(dimensions_array, axis=0)}")
