import os
import nibabel as nib
import numpy as np


def load_nifti(file_path):
    return nib.load(file_path)


def save_nifti(data, affine, header, file_path):
    new_img = nib.Nifti1Image(data, affine, header)
    nib.save(new_img, file_path)


def apply_brain_mask(fmri_img, brain_mask_img):
    fmri_data = fmri_img.get_fdata()
    brain_mask = brain_mask_img.get_fdata()
    masked_data = fmri_data * brain_mask[..., np.newaxis]
    return masked_data


def process_subject(subject_dir, output_dir):
    func_dir = os.path.join(subject_dir, 'func')
    if not os.path.exists(func_dir):
        return

    tasks = [f for f in os.listdir(func_dir) if 'preproc' in f]
    for task_file in tasks:
        if not task_file.endswith('_preproc.nii.gz'):
            continue

        base_name = task_file.split('_preproc.nii.gz')[0]
        brain_mask_file = base_name + '_brainmask.nii.gz'

        fmri_file_path = os.path.join(func_dir, task_file)
        brain_mask_file_path = os.path.join(func_dir, brain_mask_file)

        if not os.path.exists(brain_mask_file_path):
            continue

        # Load fMRI and brain mask images
        fmri_img = load_nifti(fmri_file_path)
        brain_mask_img = load_nifti(brain_mask_file_path)

        # Apply brain mask
        masked_data = apply_brain_mask(fmri_img, brain_mask_img)

        # Save new 4D data
        output_subject_dir = os.path.join(output_dir, os.path.basename(os.path.dirname(subject_dir)),
                                          os.path.basename(subject_dir), 'func')
        os.makedirs(output_subject_dir, exist_ok=True)
        output_file_path = os.path.join(output_subject_dir, task_file)

        save_nifti(masked_data, fmri_img.affine, fmri_img.header, output_file_path)


def process_all_subjects(input_dir, output_dir):
    groups = ['SCHZ', 'HC']
    for group in groups:
        group_dir = os.path.join(input_dir, group)
        if not os.path.exists(group_dir):
            continue

        subjects = [sub for sub in os.listdir(group_dir) if sub.startswith('sub-')]
        for subject in subjects:
            subject_dir = os.path.join(group_dir, subject)
            process_subject(subject_dir, output_dir)

# Input and output directories
input_dir = '/home/orion/Geo/UCLA data/FMRIPrep/data/'
output_dir = '/home/orion/Geo/UCLA data/FMRIPrep/Applied Brain Mask/'

# Process all subjects
process_all_subjects(input_dir, output_dir)
