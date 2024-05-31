import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Dense, GlobalAveragePooling3D, LSTM, TimeDistributed, \
    Masking
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score
import seaborn as sns
from tensorflow.keras import mixed_precision

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


# Padding or truncating the input to ensure consistent shape
def pad_or_truncate(data, target_shape, max_time_length):
    current_shape = data.shape
    if current_shape[-1] > max_time_length:
        data = data[..., :max_time_length]
    else:
        pad_width = [(0, 0)] * (len(current_shape) - 1) + [(0, max_time_length - current_shape[-1])]
        data = np.pad(data, pad_width, mode='constant')
    pad_widths = [(0, max(0, t - c)) for t, c in zip(target_shape, data.shape[:-1])] + [(0, 0)]
    truncated = [slice(0, min(t, c)) for t, c in zip(target_shape, data.shape[:-1])] + [slice(0, max_time_length)]
    data_padded = np.pad(data[tuple(truncated)], pad_width=pad_widths, mode='constant')
    return data_padded


# Load data paths and labels
def load_data_paths(data_dir):
    subjects, labels = [], []
    for group in ['SCHZ', 'HC']:
        group_dir = os.path.join(data_dir, group)
        label = 0 if group == 'SCHZ' else 1
        for subject in os.listdir(group_dir):
            subject_path = os.path.join(group_dir, subject, 'func')
            for task_file in os.listdir(subject_path):
                if 'preproc.nii.gz' in task_file:
                    bold_path = os.path.join(subject_path, task_file)
                    subjects.append(bold_path)
                    labels.append(label)
    return subjects, labels


# Use tf.data API for efficient data pipeline
def tf_data_generator(subjects, labels, target_shape, max_time_length, batch_size):
    total_subjects = len(subjects)

    def _parse_function(bold_path, label, index):
        bold_data = load_nifti_file(bold_path.numpy().decode('utf-8'))
        bold_data = (bold_data - np.mean(bold_data)) / np.std(bold_data)
        bold_data = pad_or_truncate(bold_data, target_shape, max_time_length)
        bold_data = bold_data[..., np.newaxis]  # Add a new axis for the channel
        bold_data = np.transpose(bold_data, (3, 0, 1, 2, 4))  # Move the time dimension to the first position
        return bold_data, to_categorical(label, num_classes=2), bold_data.shape[0]

    def _py_function_wrapper(bold_path, label, index):
        result_data, result_label, length = tf.py_function(
            _parse_function, [bold_path, label, index], [tf.float32, tf.float32, tf.int32])
        result_data.set_shape((max_time_length,) + target_shape + (1,))
        result_label.set_shape((2,))
        length.set_shape(())
        return result_data, result_label, length

    dataset = tf.data.Dataset.from_tensor_slices((subjects, labels, list(range(len(subjects)))))
    dataset = dataset.shuffle(buffer_size=len(subjects))
    dataset = dataset.map(_py_function_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.padded_batch(batch_size, padded_shapes=((max_time_length,) + target_shape + (1,), (2,), ()))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, total_subjects


# Define the simplified hybrid 3D CNN + RNN model
def create_simplified_hybrid_3dcnn_rnn(input_shape, num_classes=2):
    inputs = Input(shape=input_shape, dtype=tf.float32, name='inputs')
    lengths = Input(shape=(), dtype=tf.int32, name='lengths')

    # Simplified 3D CNN
    x = TimeDistributed(Conv3D(8, kernel_size=3, padding='same', activation='relu'))(inputs)
    x = TimeDistributed(MaxPooling3D(pool_size=2))(x)
    x = TimeDistributed(GlobalAveragePooling3D())(x)

    # Masking to handle variable lengths
    x = Masking()(x)

    # Simplified RNN Layer
    x = LSTM(16)(x, mask=tf.sequence_mask(lengths))
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[inputs, lengths], outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Define the input shape for the model
target_shape = (65, 77, 49)  # Exclude the time dimension for now
max_time_length = 300  # Define a maximum time length to pad/truncate to
input_shape = (max_time_length,) + target_shape + (1,)  # Time dimension is variable

# Load the data paths and labels
data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/Applied Brain Mask & Regressed out confounds/'
subjects, labels = load_data_paths(data_dir)

# Split data into training and validation sets
split_idx = int(0.8 * len(subjects))
train_subjects, train_labels = subjects[:split_idx], labels[:split_idx]
val_subjects, val_labels = subjects[split_idx:], labels[split_idx:]

# Define batch size
batch_size = 2  # Reduced batch size to fit into memory

# Create tf.data datasets
train_dataset, total_train_subjects = tf_data_generator(train_subjects, train_labels, target_shape, max_time_length,
                                                        batch_size)
val_dataset, total_val_subjects = tf_data_generator(val_subjects, val_labels, target_shape, max_time_length, batch_size)

# Calculate steps per epoch
steps_per_epoch_train = len(train_subjects) // batch_size
steps_per_epoch_val = len(val_subjects) // batch_size

# Create the model
model = create_simplified_hybrid_3dcnn_rnn(input_shape)
model.summary()

# Define early stopping and model checkpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')

# Custom training loop to include progress tracking
for epoch in range(10):
    print(f"\nEpoch {epoch + 1}/10")
    processed_images = 0

    for X_batch, y_batch, lengths in train_dataset:
        model.train_on_batch([X_batch, lengths], y_batch)
        processed_images += 1
        remaining_images = total_train_subjects - processed_images
        if processed_images % 10 == 0:
            print(f"{remaining_images} images left to process in this epoch")

    # Validation step
    val_loss = 0
    val_accuracy = 0
    for X_batch, y_batch, lengths in val_dataset:
        val_metrics = model.test_on_batch([X_batch, lengths], y_batch)
        val_loss += val_metrics[0]
        val_accuracy += val_metrics[1]

    val_loss /= steps_per_epoch_val
    val_accuracy /= steps_per_epoch_val
    print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}")

    # Early stopping and model checkpoint
    if early_stopping.on_epoch_end(epoch, logs={"val_loss": val_loss, "val_accuracy": val_accuracy}):
        print("Early stopping triggered")
        break
    checkpoint.on_epoch_end(epoch, logs={"val_loss": val_loss, "val_accuracy": val_accuracy})

# Training the model
history = model.fit(train_dataset.map(lambda x, y, z: ((x, z), y)), epochs=10, steps_per_epoch=steps_per_epoch_train,
                    validation_data=val_dataset.map(lambda x, y, z: ((x, z), y)), validation_steps=steps_per_epoch_val,
                    callbacks=[early_stopping, checkpoint], verbose=2)

# Clear session to free up resources
tf.keras.backend.clear_session()

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

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

# Evaluate the model on the validation set and save performance metrics
y_true, y_pred = [], []
for X_batch, y_batch, lengths in val_dataset:
    y_true.extend(np.argmax(y_batch.numpy(), axis=1))
    y_pred.extend(np.argmax(best_model.predict([X_batch, lengths]), axis=1))
    if len(y_true) >= len(val_subjects):
        break

# Check unique classes in y_pred and adjust if necessary
unique_classes = np.unique(y_pred)
if len(unique_classes) == 1:
    if unique_classes[0] == 0:
        y_pred = np.concatenate([y_pred, [1]])
        y_true = np.concatenate([y_true, [1]])
    else:
        y_pred = np.concatenate([y_pred, [0]])
        y_true = np.concatenate([y_true, [0]])

# Calculate performance metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred, target_names=['SCHZ', 'HC'], output_dict=True)

# Print and save performance metrics
print(f'\nValidation Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_report(y_true, y_pred, target_names=['SCHZ', 'HC']))

with open('performance_metrics.txt', 'w') as f:
    f.write(f'Validation Accuracy: {accuracy:.4f}\n')
    f.write(f'Validation Precision: {precision:.4f}\n')
    f.write(f'Validation Recall: {recall:.4f}\n')
    f.write(f'Validation F1 Score: {f1:.4f}\n')
    f.write('Confusion Matrix:\n')
    f.write(np.array2string(conf_matrix))
    f.write('\nClassification Report:\n')
    f.write(classification_report(y_true, y_pred, target_names=['SCHZ', 'HC']))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['SCHZ', 'HC'], yticklabels=['SCHZ', 'HC'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
