import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.data import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns


# Function to load NIfTI files
def load_nifti_file(filepath):
    img = nib.load(filepath)
    data = img.get_fdata()
    return data


# Generator function to yield data in batches
def data_generator(data_dir, groups, batch_size):
    subjects = []
    labels = []

    for group in groups:
        group_dir = os.path.join(data_dir, group)
        label = 0 if group == 'SCHZ' else 1

        for subject in os.listdir(group_dir):
            subject_dir = os.path.join(group_dir, subject, 'func')
            for task_file in os.listdir(subject_dir):
                if 'preproc.nii.gz' in task_file:
                    bold_path = os.path.join(subject_dir, task_file)
                    subjects.append(bold_path)
                    labels.append(label)

    dataset_size = len(subjects)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    def generator():
        for start_idx in range(0, dataset_size, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            if len(batch_indices) < batch_size:
                continue  # Skip incomplete batches

            batch_subjects = [subjects[i] for i in batch_indices]
            batch_labels = [labels[i] for i in batch_indices]

            X = []
            y = []

            for bold_path, label in zip(batch_subjects, batch_labels):
                bold_data = load_nifti_file(bold_path)
                bold_data = (bold_data - np.mean(bold_data)) / np.std(bold_data)
                X.append(bold_data)
                y.append(label)

            X = np.array(X)[..., np.newaxis]  # Add channel dimension
            y = to_categorical(np.array(y), num_classes=2)  # Convert labels to categorical
            yield X, y

    return dataset_size, generator


# Define the data directory
data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/data_3d/'
batch_size = 8

# Create generators for training, validation, and test data
train_size, train_generator = data_generator(data_dir, ['SCHZ', 'HC'], batch_size)
val_size, val_generator = data_generator(data_dir, ['SCHZ', 'HC'], batch_size)
test_size, test_generator = data_generator(data_dir, ['SCHZ', 'HC'], batch_size)

# Create TensorFlow datasets from generators
train_dataset = Dataset.from_generator(train_generator, output_types=(tf.float32, tf.float32),
                                       output_shapes=((batch_size, 65, 77, 49, 1), (batch_size, 2)))
val_dataset = Dataset.from_generator(val_generator, output_types=(tf.float32, tf.float32),
                                     output_shapes=((batch_size, 65, 77, 49, 1), (batch_size, 2)))
test_dataset = Dataset.from_generator(test_generator, output_types=(tf.float32, tf.float32),
                                      output_shapes=((batch_size, 65, 77, 49, 1), (batch_size, 2)))

# Ensure batching is handled correctly
train_dataset = train_dataset.unbatch().repeat().batch(batch_size)
val_dataset = val_dataset.unbatch().repeat().batch(batch_size)
test_dataset = test_dataset.unbatch().batch(batch_size)

# Calculate steps per epoch
steps_per_epoch = train_size // batch_size
validation_steps = val_size // batch_size
test_steps = test_size // batch_size


# Define the 3D CNN model
def create_3d_cnn(input_shape):
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())

    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())

    model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Define the input shape for the model
input_shape = (65, 77, 49, 1)
model = create_3d_cnn(input_shape)
model.summary()

# Verify TensorFlow is using the GPU
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
print("Using GPU:", tf.test.gpu_device_name())

# Training the model
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps)

# Save the trained model
model.save('3d_cnn_model.h5')

# Save the training history
history_file = 'training_history.txt'
np.save(history_file, history.history)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Save plots
plt.savefig('training_metrics.png')
plt.show()

# Evaluate the model on the test set and save performance metrics
y_true = []
y_pred = []

for X_batch, y_batch in test_dataset:
    y_true.extend(np.argmax(y_batch, axis=1))
    y_pred.extend(np.argmax(model.predict(X_batch), axis=1))

# Calculate performance metrics
accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred, target_names=['SCHZ', 'HC'])

# Print and save performance metrics
print(f'Test Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

with open('performance_metrics.txt', 'w') as f:
    f.write(f'Test Accuracy: {accuracy:.4f}\n')
    f.write('Confusion Matrix:\n')
    f.write(np.array2string(conf_matrix))
    f.write('\nClassification Report:\n')
    f.write(class_report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['SCHZ', 'HC'], yticklabels=['SCHZ', 'HC'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
