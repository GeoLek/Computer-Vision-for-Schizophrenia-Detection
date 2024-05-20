import pydicom
from PIL import Image
import numpy as np
import os

def ensure_transfer_syntax(dicom_dataset):
    # Check if the dataset's file_meta exists and if TransferSyntaxUID is missing
    if not hasattr(dicom_dataset, 'file_meta') or not hasattr(dicom_dataset.file_meta, 'TransferSyntaxUID'):
        # Create a FileMetaDataset if it doesn't exist
        if not hasattr(dicom_dataset, 'file_meta'):
            dicom_dataset.file_meta = pydicom.dataset.FileMetaDataset()
        # Set a default TransferSyntaxUID (Explicit VR Little Endian)
        dicom_dataset.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

def dicom_to_png(dicom_file_path, png_file_path):
    try:
        # Load the DICOM file with force=True
        dicom_image = pydicom.dcmread(dicom_file_path, force=True)
        # Ensure TransferSyntaxUID is set
        ensure_transfer_syntax(dicom_image)

        # Access and normalize the pixel data
        image_array = dicom_image.pixel_array
        image_array = ((image_array - image_array.min()) / (image_array.max() - image_array.min())) * 255.0
        image_array = image_array.astype(np.uint8)

        # Create and save the PNG image
        pil_image = Image.fromarray(image_array)
        pil_image.save(png_file_path)
        print(f"Successfully converted {os.path.basename(dicom_file_path)} to PNG.")

    except Exception as e:
        print(f"Error processing {os.path.basename(dicom_file_path)}: {str(e)}")

def convert_directory(dicom_dir_path, png_dir_path):
    if not os.path.exists(png_dir_path):
        os.makedirs(png_dir_path)

    dicom_files = [f for f in os.listdir(dicom_dir_path) if f.endswith('.dcm')]
    for dicom_file in dicom_files:
        dicom_file_path = os.path.join(dicom_dir_path, dicom_file)
        png_file_path = os.path.join(png_dir_path, dicom_file[:-4] + '.png')
        dicom_to_png(dicom_file_path, png_file_path)


# Example usage
dicom_dir_path = '/home/orion/Geo/Projects/Computer-Vision-for-Schizophrenia-Detection/Datasets/COBRE2/dmn'
png_dir_path = '/home/orion/Geo/Projects/Computer-Vision-for-Schizophrenia-Detection/Datasets/COBRE2/dmn_png'
convert_directory(dicom_dir_path, png_dir_path)
