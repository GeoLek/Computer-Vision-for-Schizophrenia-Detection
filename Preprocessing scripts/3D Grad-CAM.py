import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
import tensorflow as tf
from scipy.ndimage import zoom

# Function to load NIfTI file
def load_nifti_file(filepath):
    img = nib.load(filepath)
    data = img.get_fdata()
    return img, data

# Function to compute Grad-CAM for 3D data
def compute_gradcam_3d(model, volume, layer_name, class_idx):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(volume, axis=0))
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

# Function to resize heatmap to match volume dimensions
def resize_heatmap(heatmap, target_shape):
    factors = [t / s for t, s in zip(target_shape, heatmap.shape)]
    heatmap_resized = zoom(heatmap, factors, order=1)
    return heatmap_resized

# Function to save Grad-CAM heatmap as a NIfTI file and visualize it
def save_and_visualize_gradcam(volume, heatmap, affine, output_dir, output_filename, alpha=0.4):
    # Resize heatmap to match volume dimensions
    heatmap_resized = resize_heatmap(heatmap, volume.shape)

    # Save the heatmap as a NIfTI file
    heatmap_img = nib.Nifti1Image(heatmap_resized, affine)
    heatmap_file = os.path.join(output_dir, f"{output_filename}_gradcam.nii.gz")
    nib.save(heatmap_img, heatmap_file)
    print(f"Saved Grad-CAM heatmap to {heatmap_file}")

    # Visualize and save the heatmap overlaid on the original image
    output_file = os.path.join(output_dir, f"{output_filename}_gradcam.png")
    display = plotting.plot_anat(nib.Nifti1Image(volume, affine),
                                 title=f"{output_filename} - Grad-CAM")
    display.add_overlay(nib.Nifti1Image(heatmap_resized, affine), cmap='jet', alpha=alpha)
    display.savefig(output_file)
    display.close()
    print(f"Saved Grad-CAM visualization for {output_filename} to {output_file}")

# Load your model and volume
model_path = '/home/orion/Geo/Projects/Computer-Vision-for-Schizophrenia-Detection/Methodologies/Methodology 4: Apply 3D Brain Mask to 4D Preproc. Regress out confounds. Convert 4D Data to 3D Data/Simple 3D CNN_SMOTE_Data Augmentation_Early Stopping_ModelCheckpoint/best_model_fold_1.h5'
volume_path = '/home/orion/Geo/UCLA data/FMRIPrep/Brain Mask_Confounds_3D Converted/SCHZ/sub-50004/func/sub-50004_task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'
output_dir = '/home/orion/Geo/UCLA data/FMRIPrep/visualization-output/'

# Create output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load model and volume
model = tf.keras.models.load_model(model_path)
img, volume = load_nifti_file(volume_path)
volume = (volume - np.mean(volume)) / np.std(volume)  # Normalize the volume
affine = img.affine

# Print layer names to verify
for layer in model.layers:
    print(layer.name)

# Compute Grad-CAM
layer_name = 'conv3d_19'  # Replace with the actual layer name from your model
class_idx = np.argmax(model.predict(np.expand_dims(volume, axis=0)))
heatmap = compute_gradcam_3d(model, volume, layer_name, class_idx)

# Visualize and save Grad-CAM
output_filename = 'sub-50004_task-rest_bold_space-MNI152NLin2009cAsym_preproc'
save_and_visualize_gradcam(volume, heatmap, affine, output_dir, output_filename)

print("Finished processing the volume.")
