#Import necessary libraries
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Define the filepath to your NIfTI scan
scanFilePath = '/home/orion/Geo/Projects/Computer-Vision-for-Schizophrenia-Detection/Datasets/HAD/Hippocampus/imagesTs/test1.nii'

# Load the scan and extract data using nibabel
scan = nib.load(scanFilePath)
scanArray = scan.get_fdata()

#Get and print the scan's shape
scanArrayShape = scanArray.shape
print('The scan data array has the shape: ', scanArrayShape)