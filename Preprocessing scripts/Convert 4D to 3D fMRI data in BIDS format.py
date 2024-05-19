import os
import numpy as np
import nibabel as nib


def load_nifti_file(filepath):
    img = nib.load(filepath)
    data = img.get_fdata()
    return data, img.affine, img.header


def save_nifti_file(data, affine, header, filepath):
    img = nib.Nifti1Image(data, affine, header)
    nib.save(img, filepath)


def extract_3d_from_4d(data):
    # Take the mean across the time dimension
    return np.mean(data, axis=3)


def process_and_save(data_dir, new_data_dir):
    for group in ['SCHZ', 'HC']:
        group_dir = os.path.join(data_dir, group)
        new_group_dir = os.path.join(new_data_dir, group)
        os.makedirs(new_group_dir, exist_ok=True)

        for subject in os.listdir(group_dir):
            subject_dir = os.path.join(group_dir, subject, 'func')
            new_subject_dir = os.path.join(new_group_dir, subject, 'func')
            os.makedirs(new_subject_dir, exist_ok=True)

            for task_file in os.listdir(subject_dir):
                if 'preproc.nii.gz' in task_file:
                    bold_path = os.path.join(subject_dir, task_file)
                    new_bold_path = os.path.join(new_subject_dir, task_file)

                    # Load the 4D data
                    bold_data, affine, header = load_nifti_file(bold_path)

                    # Extract the 3D part
                    bold_data_3d = extract_3d_from_4d(bold_data)

                    # Save the new 3D data
                    save_nifti_file(bold_data_3d, affine, header, new_bold_path)
                    print(f"Saved 3D data to {new_bold_path}")


# Define the original and new data directories
data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/data/'
new_data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/data_3d/'

# Process and save the data
process_and_save(data_dir, new_data_dir)
