import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Dense, GlobalAveragePooling3D, LSTM, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt

# Set mixed precision policy
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Ensure GPU memory growth is enabled
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

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
        return bold_data, to_categorical(label, num_classes=2)

    def _py_function_wrapper(bold_path, label):
        result_data, result_label = tf.py_function(
            _parse_function, [bold_path, label], [tf.float32, tf.float32])
        result_data.set_shape((300, 65, 77, 49, 1))  # Set the shape explicitly
        result_label.set_shape((2,))
        return result_data, result_label

    dataset = tf.data.Dataset.from_tensor_slices((subjects, labels))
    dataset = dataset.shuffle(buffer_size=len(subjects))
    dataset = dataset.map(_py_function_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, total_subjects

# Define a simpler 3D CNN model
def create_simple_3dcnn(input_shape, num_classes=2):
    inputs = Input(shape=input_shape, dtype=tf.float32, name='inputs')

    # Simplified 3D CNN
    x = TimeDistributed(Conv3D(4, kernel_size=3, padding='same', activation='relu'))(inputs)
    x = TimeDistributed(MaxPooling3D(pool_size=2))(x)
    x = TimeDistributed(GlobalAveragePooling3D())(x)

    x = LSTM(8)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the input shape for the model
input_shape = (300, 65, 77, 49, 1)  # Time dimension is 300, and the 3D dimensions are fixed

# Load the data paths and labels
train_data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/Applied Brain Mask & Regressed out confounds_4D final_splitted/train'
val_data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/Applied Brain Mask & Regressed out confounds_4D final_splitted/val'

train_subjects, train_labels = load_data_paths(train_data_dir)
val_subjects, val_labels = load_data_paths(val_data_dir)

# Print class distribution
print("Train class distribution: SCHZ - {}, HC - {}".format(train_labels.count(0), train_labels.count(1)))
print("Validation class distribution: SCHZ - {}, HC - {}".format(val_labels.count(0), val_labels.count(1)))

# Define batch size
batch_size = 1  # Reduced batch size to fit into memory

# Create tf.data datasets
train_dataset, total_train_subjects = tf_data_generator(train_subjects, train_labels, batch_size)
val_dataset, total_val_subjects = tf_data_generator(val_subjects, val_labels, batch_size)

# Create the model
model = create_simple_3dcnn(input_shape)
model.summary()

# Define model checkpoint callback
checkpoint = tf.keras.callbacks.ModelCheckpoint('3dcnnlstm.h5', monitor='val_loss', save_best_only=True, mode='min')

# Train the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=10, callbacks=[checkpoint])

# Save plots for training & validation accuracy and loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.savefig('training_metrics.png')
plt.show()