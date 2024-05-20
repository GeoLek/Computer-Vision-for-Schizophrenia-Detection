import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization, Add, Activation, GlobalAveragePooling3D, multiply, Lambda
from tensorflow.data import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from scipy.ndimage import rotate, shift
import tensorflow.keras.backend as K
from sklearn.utils.class_weight import compute_class_weight
from keras.src.layers import Reshape
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


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
    shifts = np.random.uniform(-5, 5, size=3)
    volume = shift(volume, shift=shifts)
    return volume

def apply_augmentations(volume):
    if np.random.rand() > 0.5:
        volume = random_rotation(volume)
    if np.random.rand() > 0.5:
        volume = random_shift(volume)
    return volume

# Generator function to yield data in batches
def data_generator(data_dir, groups, batch_size, augment=False):
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
                if augment:
                    bold_data = apply_augmentations(bold_data)
                X.append(bold_data)
                y.append(label)

            X = np.array(X)[..., np.newaxis]  # Add channel dimension
            y = to_categorical(np.array(y), num_classes=2)  # Convert labels to categorical
            yield X, y

    return dataset_size, generator, labels

# Define the data directory
data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/data_3d/'
batch_size = 8

# Create generators for training, validation, and test data with augmentation for training
train_size, train_generator, train_labels = data_generator(data_dir, ['SCHZ', 'HC'], batch_size, augment=True)
val_size, val_generator, _ = data_generator(data_dir, ['SCHZ', 'HC'], batch_size)
test_size, test_generator, _ = data_generator(data_dir, ['SCHZ', 'HC'], batch_size)

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

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(class_weights))
print(f"Class weights: {class_weights}")

# Define channel attention block
def channel_attention(input_tensor, reduction_ratio=16):
    filters = input_tensor.shape[-1]
    se_shape = (1, 1, 1, filters)

    se = GlobalAveragePooling3D()(input_tensor)
    se = Reshape(se_shape)(se)
    se = Dense(filters // reduction_ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)

    x = multiply([input_tensor, se])
    return x

# Define spatial attention block
def spatial_attention(input_tensor):
    kernel_size = 7
    avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(input_tensor)
    max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(input_tensor)
    concat = tf.concat([avg_pool, max_pool], axis=-1)
    x = Conv3D(1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid')(concat)
    x = multiply([input_tensor, x])
    return x

# Define attention residual block
def attention_resnet_block(input_tensor, filters, kernel_size=3, strides=1):
    x = Conv3D(filters, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    x = channel_attention(x)
    x = spatial_attention(x)

    shortcut = Conv3D(filters, kernel_size=1, strides=strides, padding='same')(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# Define the 3D ResNet with Attention model
def create_3d_attention_resnet(input_shape, num_classes=2):
    inputs = Input(shape=input_shape)
    x = Conv3D(32, kernel_size=3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=3, strides=2, padding='same')(x)

    for filters in [64, 128, 256]:
        x = attention_resnet_block(x, filters)

    x = GlobalAveragePooling3D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the input shape for the model
input_shape = (65, 77, 49, 1)
model = create_3d_attention_resnet(input_shape)
model.summary()

# Verify TensorFlow is using the GPU
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
print("Using GPU:", tf.test.gpu_device_name())

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training the model
history = model.fit(train_dataset, epochs=50, validation_data=val_dataset, steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps, class_weight=class_weights, callbacks=[early_stopping])

# Save the trained model
model.save('3D ResNet CNN with attention mechanism_automatic weight calculation.h5')

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
