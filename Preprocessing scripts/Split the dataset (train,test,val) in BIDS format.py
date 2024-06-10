import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import nibabel as nib


def load_nifti_file(filepath):
    img = nib.load(filepath)
    data = img.get_fdata()
    return data


# Load data paths and labels
def load_data_paths(data_dir):
    subjects, labels = [], []
    for group in ['SCHZ', 'HC']:
        group_dir = os.path.join(data_dir, group)
        label = 0 if group == 'SCHZ' else 1
        for subject in os.listdir(group_dir):
            subject_path = os.path.join(group_dir, subject, 'func')
            for task_file in os.listdir(subject_path):
                if 'preproc.nii.gz' in task_file:
                    bold_path = os.path.join(subject_path, task_file)
                    subjects.append(bold_path)
                    labels.append(label)
    return subjects, labels


# Split the data
def split_data(subjects, labels, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    assert train_size + val_size + test_size == 1.0, "Train, validation and test sizes must sum to 1."

    # First split to get train and temp sets
    subjects_train, subjects_temp, labels_train, labels_temp = train_test_split(
        subjects, labels, train_size=train_size, stratify=labels, random_state=random_state)

    # Then split temp set into validation and test sets
    val_size_adjusted = val_size / (val_size + test_size)
    subjects_val, subjects_test, labels_val, labels_test = train_test_split(
        subjects_temp, labels_temp, train_size=val_size_adjusted, stratify=labels_temp, random_state=random_state)

    return (subjects_train, labels_train), (subjects_val, labels_val), (subjects_test, labels_test)


# Create new directory and copy files
def create_split_directories(output_dir, data_dir, train_data, val_data, test_data):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    split_data = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    for split, data in split_data.items():
        for path, label in zip(*data):
            # Calculate the relative path from the original data directory
            rel_path = os.path.relpath(path, start=data_dir)
            # Create the new destination path preserving the directory structure
            dest_path = os.path.join(output_dir, split, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(path, dest_path)


def main():
    data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/Applied Brain Mask & Regressed out confounds_4D final/'  # Replace with your data directory
    output_dir = '/home/orion/Geo/UCLA data/FMRIPrep/Applied Brain Mask & Regressed out confounds_4D final_splitted/'  # Replace with your desired output directory

    subjects, labels = load_data_paths(data_dir)
    train_data, val_data, test_data = split_data(subjects, labels)
    create_split_directories(output_dir, data_dir, train_data, val_data, test_data)


if __name__ == "__main__":
    main()