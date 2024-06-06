import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization, TimeDistributed, LSTM, GlobalAveragePooling3D
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from scipy.ndimage import rotate, shift, zoom
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
from collections import defaultdict
from tensorflow.keras import mixed_precision

# Enable mixed precision training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Define parameters
target_shape = (32, 38, 24)
max_time_length = 300
batch_size = 8

# Function to load NIfTI files
def load_nifti_file(filepath):
    img = nib.load(filepath)
    data = img.get_fdata()
    return data

# Function to resize the volume to the target shape
def resize_volume(img, target_shape):
    if img.ndim != 3:
        raise ValueError(f"Expected 3D volume, but got volume with shape {img.shape}")
    factors = [float(t) / float(s) for t, s in zip(target_shape, img.shape)]
    img_resized = zoom(img, factors, order=1)  # Use spline interpolation of order 1 (linear)
    return img_resized

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
    augmented = False
    if np.random.rand() > 0.5:
        volume = random_rotation(volume)
        augmented = True
    if np.random.rand() > 0.5:
        volume = random_shift(volume)
        augmented = True
    return volume, augmented

# Define the simplified hybrid 3D CNN + RNN model
def create_simplified_hybrid_3dcnn_rnn(input_shape, num_classes=2):
    inputs = Input(shape=input_shape, dtype=tf.float32, name='inputs')

    # First convolutional block
    x = TimeDistributed(Conv3D(16, kernel_size=3, padding='same', activation='relu'))(inputs)
    x = TimeDistributed(MaxPooling3D(pool_size=2))(x)
    x = TimeDistributed(BatchNormalization())(x)

    # Second convolutional block
    x = TimeDistributed(Conv3D(32, kernel_size=3, padding='same', activation='relu'))(x)
    x = TimeDistributed(MaxPooling3D(pool_size=2))(x)
    x = TimeDistributed(BatchNormalization())(x)

    # Third convolutional block
    x = TimeDistributed(Conv3D(64, kernel_size=3, padding='same', activation='relu'))(x)
    x = TimeDistributed(MaxPooling3D(pool_size=2))(x)
    x = TimeDistributed(BatchNormalization())(x)

    # Fourth convolutional block
    x = TimeDistributed(Conv3D(128, kernel_size=3, padding='same', activation='relu'))(x)
    x = TimeDistributed(MaxPooling3D(pool_size=2))(x)
    x = TimeDistributed(BatchNormalization())(x)

    # Instead of another max pooling layer, use global average pooling
    x = TimeDistributed(GlobalAveragePooling3D())(x)

    # LSTM layer
    x = LSTM(32, return_sequences=True)(x)
    x = LSTM(32)(x)

    # Fully connected layer with dropout
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model



# Define the input shape for the model
input_shape = (max_time_length, *target_shape, 1)

# Initialize lists to store metrics across folds
accuracies = []
precisions = []
recalls = []
f1s = []
conf_matrices = []
class_reports = []

# Function to pad sequences manually
def pad_sequences_custom(sequences, maxlen, target_shape, dtype='float32'):
    padded = np.zeros((len(sequences), maxlen, *target_shape), dtype=dtype)
    for i, seq in enumerate(sequences):
        length = min(len(seq), maxlen)
        padded[i, :length] = seq[:length]
    return padded

# Load data into numpy arrays
subjects, labels = [], []
data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/4D_Downsized/'

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

# Generator to yield file paths and labels in batches
def file_path_generator(subjects, labels, batch_size):
    dataset_size = len(subjects)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    def generator():
        while True:
            for start_idx in range(0, dataset_size, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                if len(batch_indices) < batch_size:
                    continue  # Skip incomplete batches

                batch_subjects = [subjects[i] for i in batch_indices]
                batch_labels = [labels[i] for i in batch_indices]
                yield batch_subjects, batch_labels

    return generator

# Main training and evaluation loop
num_batches = len(subjects) // batch_size
file_gen = file_path_generator(subjects, labels, batch_size)

# Load and preprocess all data in batches
for batch_no, (batch_subjects, batch_labels) in enumerate(file_gen()):
    if batch_no >= num_batches:
        break  # Stop after processing all batches

    print(f'\nProcessing batch {batch_no + 1}/{num_batches}...\n')

    X_batch = []
    y_batch = []
    for bold_path, label in zip(batch_subjects, batch_labels):
        bold_data = load_nifti_file(bold_path)
        if bold_data.ndim == 4:  # If the data has a time dimension
            bold_data = np.mean(bold_data, axis=-1)  # Take the mean across the time dimension
        bold_data = resize_volume(bold_data, target_shape)  # Resize the volume to the target shape
        bold_data, _ = apply_augmentations(bold_data)  # Apply augmentations
        bold_data = (bold_data - np.mean(bold_data)) / np.std(bold_data)
        X_batch.append(bold_data)
        y_batch.append(label)

    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)

    # Ensure batch has data from both classes before applying SMOTE
    unique_classes, counts = np.unique(y_batch, return_counts=True)
    if len(unique_classes) < 2 or min(counts) < 2:
        print(f'Skipping batch {batch_no + 1} due to insufficient class samples')
        continue  # Skip batch if it contains only one class or not enough samples for SMOTE

    # Apply SMOTE to each batch
    X_reshaped = X_batch.reshape((X_batch.shape[0], -1))  # Flatten data
    min_class_samples = min(counts)
    k_neighbors = min(5, min_class_samples - 1)  # Ensure k_neighbors is less than the minimum class samples
    smote = SMOTE(k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X_reshaped, y_batch)
    X_resampled = X_resampled.reshape((-1, *target_shape))

    # Pad sequences to the required length (max_time_length)
    X_resampled_padded = pad_sequences_custom(X_resampled, maxlen=max_time_length, target_shape=target_shape)

    # Ensure padding does not add extra dimension
    X_resampled_padded = np.squeeze(X_resampled_padded)

    # Split batch data into training (64%), validation (16%), and testing (20%) sets
    split_idx_train = int(0.64 * len(X_resampled))
    split_idx_val = int(0.80 * len(X_resampled))

    X_train, X_val, X_test = X_resampled_padded[:split_idx_train], X_resampled_padded[split_idx_train:split_idx_val], X_resampled_padded[split_idx_val:]
    y_train, y_val, y_test = y_resampled[:split_idx_train], y_resampled[split_idx_train:split_idx_val], y_resampled[split_idx_val:]

    X_train = X_train[..., np.newaxis]  # Add channel dimension
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    y_train = to_categorical(y_train, num_classes=2)
    y_val = to_categorical(y_val, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    model = create_simplified_hybrid_3dcnn_rnn((max_time_length, *target_shape, 1))
    model.summary()

    # Define early stopping and model checkpoint callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f'best_model_batch_{batch_no + 1}.h5', monitor='val_loss', save_best_only=True, mode='min')

    # Training the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=batch_size, validation_data=(X_val, y_val),  # Use manual validation data
                        callbacks=[early_stopping, checkpoint], verbose=2)

    # Load the best model for evaluation
    best_model = tf.keras.models.load_model(f'best_model_batch_{batch_no + 1}.h5')

    # Evaluate the model on the testing set and save performance metrics
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(best_model.predict(X_test), axis=1)

    # Check unique classes in y_pred and adjust if necessary
    unique_classes = np.unique(y_pred)
    if len(unique_classes) == 1:
        if unique_classes[0] == 0:
            y_pred = np.concatenate([y_pred, [1]])
            y_true = np.concatenate([y_true, [1]])
        else:
            y_pred = np.concatenate([y_pred, [0]])
            y_true = np.concatenate([y_true, [0]])

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrices.append(conf_matrix)
    class_report = classification_report(y_true, y_pred, target_names=['SCHZ', 'HC'], output_dict=True, zero_division=0)
    class_reports.append(class_report)

    # Print and save performance metrics
    print(f'\nBatch {batch_no + 1} - Testing Accuracy: {accuracy:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(classification_report(y_true, y_pred, target_names=['SCHZ', 'HC'], zero_division=0))

    with open(f'performance_metrics_batch_{batch_no + 1}.txt', 'w') as f:
        f.write(f'Batch {batch_no + 1} - Testing Accuracy: {accuracy:.4f}\n')
        f.write(f'Batch {batch_no + 1} - Testing Precision: {precision:.4f}\n')
        f.write(f'Batch {batch_no + 1} - Testing Recall: {recall:.4f}\n')
        f.write(f'Batch {batch_no + 1} - Testing F1 Score: {f1:.4f}\n')
        f.write('Confusion Matrix:\n')
        f.write(np.array2string(conf_matrix))
        f.write('\nClassification Report:\n')
        f.write(classification_report(y_true, y_pred, target_names=['SCHZ', 'HC'], zero_division=0))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['SCHZ', 'HC'], yticklabels=['SCHZ', 'HC'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - Batch {batch_no + 1}')
    plt.savefig(f'confusion_matrix_batch_{batch_no + 1}.png')
    plt.show()

# Calculate and print the average metrics across all batches
average_accuracy = np.mean(accuracies)
average_precision = np.mean(precisions)
average_recall = np.mean(recalls)
average_f1 = np.mean(f1s)
average_conf_matrix = np.mean(conf_matrices, axis=0)
average_conf_matrix_rounded = np.rint(average_conf_matrix).astype(int)

# Function to accumulate classification report
def accumulate_classification_report(reports):
    total_reports = len(reports)
    avg_report = defaultdict(lambda: defaultdict(float))
    total_support = defaultdict(int)  # Dictionary to accumulate support values

    for report in reports:
        for label, metrics in report.items():
            if isinstance(metrics, dict):  # Ensure metrics is a dictionary
                for metric, value in metrics.items():
                    avg_report[label][metric] += value
                total_support[label] += metrics['support']  # Accumulate total support

    for label, metrics in avg_report.items():
        for metric in metrics:
            if metric != 'support':
                avg_report[label][metric] /= total_reports
        avg_report[label]['support'] = total_support[label]  # Use total support

    return avg_report

# Calculate the average classification report
average_classification_report = accumulate_classification_report(class_reports)

# Print the average classification report
print("Average Classification Report:")
print(f"{'Label':<15}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}{'Support':<10}")
for label, metrics in average_classification_report.items():
    if isinstance(metrics, dict):  # Ensure metrics is a dictionary
        print(f"{label:<15}{metrics['precision']:<10.2f}{metrics['recall']:<10.2f}{metrics['f1-score']:<10.2f}{int(metrics['support']):<10}")

# Write the average classification report to the file
with open('average_performance_metrics.txt', 'w') as f:
    f.write(f'Average Testing Accuracy across all batches: {average_accuracy:.4f}\n')
    f.write(f'Average Testing Precision across all batches: {average_precision:.4f}\n')
    f.write(f'Average Testing Recall across all batches: {average_recall:.4f}\n')
    f.write(f'Average Testing F1 Score across all batches: {average_f1:.4f}\n')
    f.write('Average Confusion Matrix:\n')
    f.write(np.array2string(average_conf_matrix_rounded))
    f.write('\nAverage Classification Report:\n')
    f.write(f"{'Label':<15}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}{'Support':<10}\n")
    for label, metrics in average_classification_report.items():
        if isinstance(metrics, dict):  # Ensure metrics is a dictionary
            f.write(f"{label:<15}{metrics['precision']:<10.2f}{metrics['recall']:<10.2f}{metrics['f1-score']:<10.2f}{int(metrics['support']):<10}\n")

# Plot average confusion matrix with the sum of all batches
plt.figure(figsize=(8, 6))
sns.heatmap(average_conf_matrix_rounded, annot=True, fmt='d', cmap='Blues', xticklabels=['SCHZ', 'HC'], yticklabels=['SCHZ', 'HC'],
            cbar_kws={'label': 'Count'}, annot_kws={"size": 14, "color": 'black'})
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Average Confusion Matrix')
plt.savefig('average_confusion_matrix.png')
plt.show()
