import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling3D, Add
from tensorflow.data import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from scipy.ndimage import rotate, shift
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Function to load NIfTI files
def load_nifti_file(filepath):
    img = nib.load(filepath)
    data = img.get_fdata()
    return data

# Custom augmentation functions
def random_rotation(volume):
    angles = np.random.uniform(-10, 10, size=3)
    volume = rotate(volume, angle=angles[0], axes=(1, 2), reshape=False)
    volume = rotate(volume, angle=angles[1], axes=(0, 2), reshape=False)
    volume = rotate(volume, angle=angles[2], axes=(0, 1), reshape=False)
    return volume

def random_shift(volume):
    shifts = np.random.uniform(-5, 5, size=volume.ndim)
    volume = shift(volume, shift=shifts)
    return volume

def apply_augmentations(volume):
    if np.random.rand() > 0.5:
        volume = random_rotation(volume)
    if np.random.rand() > 0.5:
        volume = random_shift(volume)
    return volume

# Generator function to yield data in batches
def data_generator(data, labels, batch_size, augment=False):
    dataset_size = len(data)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    def generator():
        while True:
            for start_idx in range(0, dataset_size, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                if len(batch_indices) < batch_size:
                    continue  # Skip incomplete batches

                X = []
                y = []

                for i in batch_indices:
                    bold_data = data[i]
                    label = labels[i]
                    if augment:
                        bold_data = apply_augmentations(bold_data)
                    X.append(bold_data)
                    y.append(label)

                X = np.array(X)[..., np.newaxis]  # Add channel dimension
                y = to_categorical(np.array(y), num_classes=2)  # Convert labels to categorical
                yield X, y

    return generator

# Load data into numpy arrays
subjects, labels = [], []
data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/Applied Brain Mask/3D Converted/'
batch_size = 8
for group in ['SCHZ', 'HC']:
    group_dir = os.path.join(data_dir, group)
    label = 0 if group == 'SCHZ' else 1
    for subject in os.listdir(group_dir):
        subject_dir = os.path.join(group_dir, subject, 'func')
        for task_file in os.listdir(subject_dir):
            if 'preproc.nii.gz' in task_file:
                bold_path = os.path.join(subject_dir, task_file)
                subjects.append(bold_path)
                labels.append(label)

X, y = [], []
for bold_path, label in zip(subjects, labels):
    bold_data = load_nifti_file(bold_path)
    bold_data = (bold_data - np.mean(bold_data)) / np.std(bold_data)
    X.append(bold_data)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Reshape data for SMOTE
X_reshaped = X.reshape((X.shape[0], -1))  # Flatten data

# Apply SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)

# Reshape data back to original shape
X_resampled = X_resampled.reshape((-1, 65, 77, 49))
# Split data back into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Convert to TensorFlow datasets using data generators
train_gen = data_generator(X_train, y_train, batch_size, augment=True)
val_gen = data_generator(X_val, y_val, batch_size)

train_dataset = Dataset.from_generator(train_gen, output_types=(tf.float32, tf.float32),
                                       output_shapes=((batch_size, 65, 77, 49, 1), (batch_size, 2)))
val_dataset = Dataset.from_generator(val_gen, output_types=(tf.float32, tf.float32),
                                     output_shapes=((batch_size, 65, 77, 49, 1), (batch_size, 2)))

# Define a residual block
def resnet_block(input_tensor, filters, kernel_size=3, strides=1):
    x = Conv3D(filters, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    shortcut = Conv3D(filters, kernel_size=1, strides=strides, padding='same')(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# Define the 3D ResNet model
def create_3d_resnet(input_shape, num_classes=2):
    inputs = Input(shape=input_shape)
    x = Conv3D(32, kernel_size=3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=3, strides=2, padding='same')(x)

    for filters in [64, 128, 256]:
        x = resnet_block(x, filters)

    x = GlobalAveragePooling3D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the input shape for the model
input_shape = (65, 77, 49, 1)
model = create_3d_resnet(input_shape)
model.summary()

# Verify TensorFlow is using the GPU
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
print("Using GPU:", tf.test.gpu_device_name())

# Define early stopping and model checkpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')

# Training the model
history = model.fit(train_dataset, epochs=50, validation_data=val_dataset, steps_per_epoch=len(X_train) // batch_size,
                    validation_steps=len(X_val) // batch_size, callbacks=[early_stopping, checkpoint])

# Save the training history as a text file
history_file = 'training_history.txt'
with open(history_file, 'w') as f:
    for epoch in range(len(history.history['accuracy'])):
        f.write(f"Epoch {epoch + 1}\n")
        f.write(f"accuracy: {history.history['accuracy'][epoch]}\n")
        f.write(f"val_accuracy: {history.history['val_accuracy'][epoch]}\n")
        f.write(f"loss: {history.history['loss'][epoch]}\n")
        f.write(f"val_loss: {history.history['val_loss'][epoch]}\n")
        f.write("\n")

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

# Load the best model for evaluation
best_model = tf.keras.models.load_model('best_model.h5')

# Evaluate the model on the test set and save performance metrics
test_gen = data_generator(X, y, batch_size)  # Generate the original test data
test_dataset = Dataset.from_generator(test_gen, output_types=(tf.float32, tf.float32),
                                      output_shapes=((batch_size, 65, 77, 49, 1), (batch_size, 2)))

y_true = []
y_pred = []

for X_batch, y_batch in test_dataset.take(len(X) // batch_size):
    y_true.extend(np.argmax(y_batch, axis=1))
    y_pred.extend(np.argmax(best_model.predict(X_batch), axis=1))

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