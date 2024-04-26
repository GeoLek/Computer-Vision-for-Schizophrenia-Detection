import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Verify if TensorFlow is using GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Data Loader
def load_nifti_file(file_path, scale=True):
    img = nib.load(file_path)
    data = img.get_fdata()
    if scale:
        # Normalize the image data to 0-1
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = np.expand_dims(data, axis=-1)  # Add channel dimension
    return data.astype(np.float32)

def prepare_dataset(paths, labels, batch_size=2):
    def generator():
        for path, label in zip(paths, labels):
            img = load_nifti_file(path)
            yield img, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.int32),
        output_shapes=((65, 77, 49, 1), ())  # Adjusted to match your data
    )
    return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Define model
def build_model():
    model = models.Sequential([
        layers.InputLayer(input_shape=(65, 77, 49, 1)),
        layers.Conv3D(32, (3, 3, 3), activation='relu'),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Conv3D(64, (3, 3, 3), activation='relu'),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

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

# Set up dataset paths for train and validation
train_base = '/home/orion/Geo/Projects/Computer-Vision-for-Schizophrenia-Detection/Working_dataset/train/'
val_base = '/home/orion/Geo/Projects/Computer-Vision-for-Schizophrenia-Detection/Working_dataset/val/'

# Get paths and labels for training and validation datasets
train_paths, train_labels = get_paths_and_labels(train_base, ['HC', 'SCHZ'])
val_paths, val_labels = get_paths_and_labels(val_base, ['HC', 'SCHZ'])

# Prepare datasets for training and validation
train_dataset = prepare_dataset(train_paths, train_labels, batch_size=2) # You can adjust batch size
val_dataset = prepare_dataset(val_paths, val_labels, batch_size=2) # You can adjust batch size

# Build the model
model = build_model()

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    verbose=1
)

# Save the trained model
model.save('3d_cnn_model.h5')

# Plot and save the validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('validation_loss.png')
plt.show()