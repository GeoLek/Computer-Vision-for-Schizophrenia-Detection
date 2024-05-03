import os
import nibabel as nib

def get_paths_and_labels(base_dir, categories):
    paths, labels = [], []
    label_dict = {'HC': 0, 'SCHZ': 1}

    for category in categories:
        cat_path = os.path.join(base_dir, category)
        for entry in os.listdir(cat_path):
            entry_path = os.path.join(cat_path, entry)
            if os.path.isfile(entry_path) and entry_path.endswith('.nii.gz'):
                paths.append(entry_path)
                labels.append(label_dict[category])
    return paths, labels

def list_file_dimensions(paths):
    """List dimensions of each NIfTI file."""
    for path in paths:
        img = nib.load(path)
        print(f"File: {path}, Dimension: {img.shape}")

# Set up dataset paths for train, validation, and test
train_base = '/home/orion/Geo/Projects/Computer-Vision-for-Schizophrenia-Detection/Working_dataset/train/'
val_base = '/home/orion/Geo/Projects/Computer-Vision-for-Schizophrenia-Detection/Working_dataset/val/'
test_base = '/home/orion/Geo/Projects/Computer-Vision-for-Schizophrenia-Detection/Working_dataset/test/'

# Retrieve paths for training, validation, and testing datasets
train_paths, train_labels = get_paths_and_labels(train_base, ['HC', 'SCHZ'])
val_paths, val_labels = get_paths_and_labels(val_base, ['HC', 'SCHZ'])
test_paths, test_labels = get_paths_and_labels(test_base, ['HC', 'SCHZ'])

# List dimensions for each set
print("Training Data Dimensions:")
list_file_dimensions(train_paths)

print("\nValidation Data Dimensions:")
list_file_dimensions(val_paths)

print("\nTesting Data Dimensions:")
list_file_dimensions(test_paths)
