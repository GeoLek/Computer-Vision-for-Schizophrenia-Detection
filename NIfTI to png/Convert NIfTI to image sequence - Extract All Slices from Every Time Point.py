import nibabel as nib
from PIL import Image
import numpy as np
import os


def nifti_to_image_sequence_all_slices(nifti_file_path, output_folder):
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

    # Iterate over every time point
    for t in range(data.shape[3]):
        # For each time point, iterate over every slice in the third dimension
        for z in range(data.shape[2]):
            # Select the z-th slice at time point t
            slice = data[:, :, z, t]

            # Convert to PIL image and save
            img = Image.fromarray(slice)
            img.save(f"{output_folder}/time_{t:03}_slice_{z:03}.png")


# Example usage
nifti_to_image_sequence_all_slices("/home/orion/Geo/Projects/Computer-Vision-for-Schizophrenia-Detection/NIfTI to Image/swarrest0.nii", "/home/orion/Geo/Projects/Computer-Vision-for-Schizophrenia-Detection/NIfTI to Image")
