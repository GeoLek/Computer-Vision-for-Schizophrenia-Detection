import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Function to compute Grad-CAM
def compute_gradcam(model, image, layer_name, class_idx):
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

# Function to display and save the heatmap on image
def display_and_save_gradcam(image, heatmap, output_path, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.expand_dims(heatmap, axis=-1)
    heatmap = tf.image.resize(heatmap, (image.shape[1], image.shape[2])).numpy()

    plt.imshow(image[0, :, :, 0], cmap='gray')
    plt.imshow(heatmap, cmap='jet', alpha=alpha)
    plt.axis('off')
    plt.show()

    # Save the image
    plt.imsave(output_path, image[0, :, :, 0], cmap='gray')
    heatmap_overlay = plt.imread(output_path)
    plt.imshow(heatmap_overlay, cmap='gray')
    plt.imshow(heatmap, cmap='jet', alpha=alpha)
    plt.axis('off')
    plt.savefig(output_path)


# Load your model and image
model = tf.keras.models.load_model('/home/orion/Geo/Projects/Computer-Vision-for-Schizophrenia-Detection/2D CNN models/best_model.h5')
image = tf.keras.preprocessing.image.load_img('/home/orion/Geo/UCLA data/FMRIPrep/Brain Mask_2D Converted/Downsampled/SCHZ/sub-50004/func/sub-50004_task-rest_bold_space-MNI152NLin2009cAsym_preproc.png', target_size=(48, 48))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = image / 255.0  # Normalize the image

# Ensure the image has the right shape (batch_size, height, width, channels)
if image.shape[-1] == 1:  # Already grayscale
    pass
elif image.shape[-1] == 3:  # Convert to grayscale if it was loaded as RGB
    image = np.mean(image, axis=-1, keepdims=True)

# Print layer names to find the correct convolutional layer
for layer in model.layers:
    print(layer.name)

# Replace 'last_conv_layer_name' with the actual name of the last convolutional layer
layer_name = 'conv2d'  # Change this to your actual last convolutional layer name
class_idx = np.argmax(model.predict(image))
heatmap = compute_gradcam(model, image, layer_name, class_idx)

# Display and save Grad-CAM
output_path = 'gradcam_result_2d.png'  # Path where the image will be saved
display_and_save_gradcam(image, heatmap, output_path)