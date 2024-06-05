import numpy as np
import tensorflow as tf
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


# Function to load NIfTI file
def load_nifti_file(filepath):
    img = nib.load(filepath)
    data = img.get_fdata()
    return data


# Function to compute Grad-CAM for 4D data
def compute_gradcam_4d(model, volume, layer_name, class_idx):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(volume, axis=0))
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3, 4))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


# Function to resize heatmap to match volume dimensions
def resize_heatmap(heatmap, target_shape):
    factors = [t / s for t, s in zip(target_shape, heatmap.shape)]
    heatmap_resized = zoom(heatmap, factors, order=1)
    return heatmap_resized


# Function to display heatmap on a 4D volume slice
def display_gradcam_4d(volume, heatmap, slice_index, time_index, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.expand_dims(heatmap, axis=-1)
    heatmap_resized = resize_heatmap(heatmap, volume.shape[:-1])

    plt.imshow(volume[slice_index, :, :, time_index], cmap='gray')
    plt.imshow(heatmap_resized[slice_index, :, :], cmap='jet', alpha=alpha)
    plt.axis('off')
    plt.show()


# Load your model and volume
model = tf.keras.models.load_model('path_to_your_4d_model.h5')
volume = load_nifti_file('path_to_your_4d_volume.nii.gz')
volume = (volume - np.mean(volume)) / np.std(volume)  # Normalize the volume

# Compute Grad-CAM
layer_name = 'last_conv4d_layer_name'  # Replace with your actual layer name
class_idx = np.argmax(model.predict(np.expand_dims(volume, axis=0)))
heatmap = compute_gradcam_4d(model, volume, layer_name, class_idx)

# Display Grad-CAM for a specific slice and time index
slice_index = volume.shape[0] // 2  # Choose the middle spatial slice
time_index = volume.shape[3] // 2  # Choose the middle time frame
display_gradcam_4d(volume, heatmap, slice_index, time_index)
