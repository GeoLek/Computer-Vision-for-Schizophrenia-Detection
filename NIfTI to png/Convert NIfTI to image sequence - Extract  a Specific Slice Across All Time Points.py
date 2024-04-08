import nibabel as nib
from PIL import Image
import numpy as np
import os


def nifti_to_specific_slice_over_time(nifti_file_path, output_folder, slice_index):
    # Load the NIfTI file
    nifti = nib.load(nifti_file_path)
    data = nifti.get_fdata()

    # Normalize the data to 0-255
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data) * 255
    data = data.astype(np.uint8)

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Determine the middle slice index if not provided
    if slice_index is None:
        slice_index = data.shape[2] // 2

    # Iterate over every time point and extract the specific slice
    for t in range(data.shape[3]):
        # Select the specific slice at time point t
        specific_slice = data[:, :, slice_index, t]

        # Convert to PIL image and save
        img = Image.fromarray(specific_slice)
        img.save(f"{output_folder}/time_{t:03}_slice_{slice_index:03}.png")


# Example usage - extracts the middle slice across all time points
nifti_to_specific_slice_over_time("/home/orion/Geo/Projects/Computer-Vision-for-Schizophrenia-Detection/NIfTI to Image/swarrest0.nii", "/home/orion/Geo/Projects/Computer-Vision-for-Schizophrenia-Detection/NIfTI to Image", None)
