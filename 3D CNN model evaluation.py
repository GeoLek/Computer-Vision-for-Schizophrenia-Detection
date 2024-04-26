import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras import models
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

# Load the saved model
model = models.load_model('3d_cnn_model.h5')

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

# Set up dataset paths for testing
test_base = '/home/orion/Geo/Projects/Computer-Vision-for-Schizophrenia-Detection/Working_dataset/test/'

# Get paths and labels for the test dataset
test_paths, test_labels = get_paths_and_labels(test_base, ['HC', 'SCHZ'])

# Prepare dataset for testing
test_dataset = prepare_dataset(test_paths, test_labels, batch_size=2)  # Batch size of 1 for evaluation

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc:.3f}")

# Predictions for confusion matrix
y_pred = model.predict(test_dataset)
y_pred = np.round(y_pred).astype(int)
y_true = np.concatenate([y for x, y in test_dataset], axis=0)

# Plot and save the confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Print and save the classification report
report = classification_report(y_true, y_pred, target_names=['Healthy', 'Schizophrenia'])
print(report)
with open('classification_report.txt', 'w') as f:
    f.write(report)
