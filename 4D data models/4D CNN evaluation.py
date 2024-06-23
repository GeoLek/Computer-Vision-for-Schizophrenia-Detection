import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import Input, MaxPooling3D, Dense, GlobalAveragePooling3D, Layer, Conv3D
import seaborn as sns
import matplotlib.pyplot as plt

class Conv4D(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(Conv4D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv3d = Conv3D(filters, kernel_size, padding='same', activation='relu')

    def call(self, inputs):
        # Iterate over the time dimension and apply Conv3D to each time slice
        time_steps = inputs.shape[1]
        output = []
        for t in range(time_steps):
            output.append(self.conv3d(inputs[:, t, ...]))
        return tf.stack(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]) + self.conv3d.compute_output_shape(input_shape[2:])

    def get_config(self):
        config = super(Conv4D, self).get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size})
        return config

# Function to load NIfTI files
def load_nifti_file(filepath):
    img = nib.load(filepath)
    data = img.get_fdata()
    return data

# Load data paths and labels
def load_data_paths(data_dir):
    subjects, labels = [], []
    for group in ['HC', 'SCHZ']:
        group_dir = os.path.join(data_dir, group)
        label = 0 if group == 'SCHZ' else 1
        for subject in os.listdir(group_dir):
            subject_dir = os.path.join(group_dir, subject, 'func')
            bold_path = os.path.join(subject_dir, 'preproc.nii.gz')
            if os.path.isfile(bold_path):
                subjects.append(bold_path)
                labels.append(label)
    return subjects, labels

# Use tf.data API for efficient data pipeline
def tf_data_generator(subjects, labels, batch_size):
    total_subjects = len(subjects)

    def _parse_function(bold_path, label):
        bold_data = load_nifti_file(bold_path.numpy().decode('utf-8'))
        bold_data = (bold_data - np.mean(bold_data)) / np.std(bold_data)
        bold_data = bold_data[..., np.newaxis]  # Add a new axis for the channel
        bold_data = np.transpose(bold_data, (3, 0, 1, 2, 4))  # Move the time dimension to the first position
        return bold_data, tf.one_hot(label, 2)

    def _py_function_wrapper(bold_path, label):
        result_data, result_label = tf.py_function(
            _parse_function, [bold_path, label], [tf.float32, tf.float32])
        result_data.set_shape((300, 65, 77, 49, 1))  # Set the shape explicitly
        result_label.set_shape((2,))
        return result_data, result_label

    dataset = tf.data.Dataset.from_tensor_slices((subjects, labels))
    dataset = dataset.map(_py_function_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, total_subjects

# Define the input shape for the model
input_shape = (300, 65, 77, 49, 1) # Time dimension is 300, and the 3D dimensions are fixed

# Load the data paths and labels
test_data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/Applied Brain Mask & Regressed out confounds_4D final_splitted/test'
subjects, labels = load_data_paths(test_data_dir)

# Print class distribution
print("Test class distribution: SCHZ - {}, HC - {}".format(labels.count(0), labels.count(1)))

# Define batch size
batch_size = 1  # Adjust based on memory

# Create tf.data dataset
test_dataset, total_test_subjects = tf_data_generator(subjects, labels, batch_size)

# Load the trained model
model = load_model('best_model.h5', custom_objects={'Conv4D': Conv4D})

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}')

# Generate predictions and calculate performance metrics
y_true = []
y_pred = []
for X_batch, y_batch in test_dataset:
    y_true_batch = np.argmax(y_batch.numpy(), axis=1)
    y_pred_batch = np.argmax(model.predict(X_batch), axis=1)
    y_true.extend(y_true_batch)
    y_pred.extend(y_pred_batch)

# Calculate confusion matrix and classification report
conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred, target_names=['SCHZ', 'HC'], output_dict=True)

# Save the classification report and confusion matrix
with open('classification_report.txt', 'w') as f:
    f.write("Classification Report:\n")
    f.write(classification_report(y_true, y_pred, target_names=['SCHZ', 'HC']))
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(conf_matrix))
    f.write("\n\nDetailed Report:\n")
    for label, metrics in class_report.items():
        if isinstance(metrics, dict):
            f.write(f"{label}:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value}\n")
            f.write("\n")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['SCHZ', 'HC'], yticklabels=['SCHZ', 'HC'], cbar_kws={'label': 'Count'}, annot_kws={"size": 14, "color": 'black'})
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
