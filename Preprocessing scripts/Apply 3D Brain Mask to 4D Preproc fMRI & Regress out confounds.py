import os
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def load_nifti(file_path):
    print(f"Loading NIfTI file: {file_path}")
    return nib.load(file_path)


def save_nifti(data, affine, header, file_path):
    print(f"Saving NIfTI file: {file_path}")
    new_img = nib.Nifti1Image(data, affine, header)
    nib.save(new_img, file_path)


def apply_brain_mask(fmri_img, brain_mask_img):
    fmri_data = fmri_img.get_fdata()
    brain_mask = brain_mask_img.get_fdata()
    masked_data = fmri_data * brain_mask[..., np.newaxis]
    return masked_data


def regress_out_confounds(fmri_data, confound_file):
    print(f"Regressing out confounds using file: {confound_file}")
    confounds = pd.read_csv(confound_file, sep='\t')
    confound_columns = ['WhiteMatter', 'GlobalSignal', 'FramewiseDisplacement', 'X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']
    confounds = confounds[confound_columns].fillna(0)

    X = confounds.values
    n_timepoints = fmri_data.shape[-1]
    cleaned_data = np.zeros_like(fmri_data)

    for i in range(fmri_data.shape[0]):
        for j in range(fmri_data.shape[1]):
            for k in range(fmri_data.shape[2]):
                y = fmri_data[i, j, k, :]
                reg = LinearRegression().fit(X, y)
                y_pred = reg.predict(X)
                cleaned_data[i, j, k, :] = y - y_pred

    return cleaned_data


def process_subject(subject_dir, output_dir):
    print(f"Processing subject directory: {subject_dir}")
    func_dir = os.path.join(subject_dir, 'func')
    if not os.path.exists(func_dir):
        print(f"Function directory does not exist: {func_dir}")
        return

    tasks = [f for f in os.listdir(func_dir) if 'preproc' in f]
    for task_file in tasks:
        if not task_file.endswith('_preproc.nii.gz'):
            continue

        base_name = task_file.split('_preproc.nii.gz')[0]
        brain_mask_file = base_name + '_brainmask.nii.gz'
        confound_file = base_name.split('_bold_space-MNI152NLin2009cAsym')[0] + '_bold_confounds.tsv'

        fmri_file_path = os.path.join(func_dir, task_file)
        brain_mask_file_path = os.path.join(func_dir, brain_mask_file)
        confound_file_path = os.path.join(func_dir, confound_file)

        if not os.path.exists(brain_mask_file_path):
            print(f"Brain mask file does not exist: {brain_mask_file_path}")
            continue
        if not os.path.exists(confound_file_path):
            print(f"Confound file does not exist: {confound_file_path}")
            continue

        # Load fMRI and brain mask images
        fmri_img = load_nifti(fmri_file_path)
        brain_mask_img = load_nifti(brain_mask_file_path)

        # Apply brain mask
        masked_data = apply_brain_mask(fmri_img, brain_mask_img)

        # Regress out confounds
        cleaned_data = regress_out_confounds(masked_data, confound_file_path)

        # Save new 4D data
        relative_subject_path = os.path.relpath(subject_dir, input_dir)
        output_subject_dir = os.path.join(output_dir, relative_subject_path, 'func')
        os.makedirs(output_subject_dir, exist_ok=True)
        output_file_path = os.path.join(output_subject_dir, task_file)

        save_nifti(cleaned_data, fmri_img.affine, fmri_img.header, output_file_path)


def process_all_subjects(input_dir, output_dir):
    groups = ['SCHZ', 'HC']
    for group in groups:
        group_dir = os.path.join(input_dir, group)
        if not os.path.exists(group_dir):
            print(f"Group directory does not exist: {group_dir}")
            continue

        subjects = [sub for sub in os.listdir(group_dir) if sub.startswith('sub-')]
        for subject in subjects:
            subject_dir = os.path.join(group_dir, subject)
            process_subject(subject_dir, output_dir)


# Input and output directories
input_dir = '/home/orion/Geo/UCLA data/FMRIPrep/data/'
output_dir = '/home/orion/Geo/UCLA data/FMRIPrep/Applied Brain Mask & Regressed out confounds/'

# Process all subjects
process_all_subjects(input_dir, output_dir)

