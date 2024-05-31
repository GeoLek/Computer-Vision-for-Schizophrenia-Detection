import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Dense, GlobalAveragePooling3D, LSTM, TimeDistributed, \
    Masking
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score
import seaborn as sns
from tensorflow.keras import mixed_precision
from imblearn.over_sampling import SMOTE
from collections import defaultdict
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import resample

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


# Manually balance the dataset by oversampling the minority class
def balance_dataset(subjects, labels):
    subjects = np.array(subjects)
    labels = np.array(labels)

    # Identify minority and majority classes
    class_counts = np.bincount(labels)
    minority_class = np.argmin(class_counts)
    majority_class = np.argmax(class_counts)

    # Separate subjects and labels into minority and majority
    minority_subjects = subjects[labels == minority_class]
    majority_subjects = subjects[labels == majority_class]
    minority_labels = labels[labels == minority_class]
    majority_labels = labels[labels == majority_class]

    # Calculate how many samples are needed to match the majority class count
    n_samples_needed = len(majority_subjects) - len(minority_subjects)

    # Resample the minority class to match the majority class count
    resampled_minority_subjects = resample(minority_subjects, replace=True, n_samples=len(majority_subjects),
                                           random_state=42)
    resampled_minority_labels = np.full(len(majority_subjects), minority_class)

    # Combine the original and resampled minority samples with the majority samples
    balanced_subjects = np.concatenate([majority_subjects, resampled_minority_subjects])
    balanced_labels = np.concatenate([majority_labels, resampled_minority_labels])

    return balanced_subjects.tolist(), balanced_labels.tolist()


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

# Print class distribution
print("Class distribution: SCHZ - {}, HC - {}".format(labels.count(0), labels.count(1)))

# Split data into training and validation sets
split_idx = int(0.8 * len(subjects))
train_subjects, train_labels = subjects[:split_idx], labels[:split_idx]
val_subjects, val_labels = subjects[split_idx:], labels[split_idx:]

# Balance the training dataset
train_subjects, train_labels = balance_dataset(train_subjects, train_labels)

# Print new class distribution after balancing
print("Class distribution after balancing: SCHZ - {}, HC - {}".format(train_labels.count(0), train_labels.count(1)))

# Define batch size
batch_size = 1  # Reduced batch size to fit into memory

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

# Define model checkpoint callback
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')

# Custom training loop to include progress tracking and metric logging
best_val_loss = float('inf')
patience = 1
wait = 0

# Store metrics for each epoch
history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
all_y_true, all_y_pred = [], []
all_conf_matrices = []
all_class_reports = []

for epoch in range(10):
    print(f"\nEpoch {epoch + 1}/10")
    processed_images = 0

    train_loss = 0
    train_accuracy = 0

    for X_batch, y_batch, lengths in train_dataset:
        train_metrics = model.train_on_batch([X_batch, lengths], y_batch)
        train_loss += train_metrics[0]
        train_accuracy += train_metrics[1]
        processed_images += batch_size
        remaining_images = total_train_subjects - processed_images
        if processed_images % (10 * batch_size) == 0:
            print(f"{remaining_images} images left to process in this epoch")

    train_loss /= steps_per_epoch_train
    train_accuracy /= steps_per_epoch_train

    # Validation step
    val_loss = 0
    val_accuracy = 0
    all_y_true_epoch = []
    all_y_pred_epoch = []

    for X_batch, y_batch, lengths in val_dataset:
        val_metrics = model.test_on_batch([X_batch, lengths], y_batch)
        val_loss += val_metrics[0]
        val_accuracy += val_metrics[1]

        y_true_batch = np.argmax(y_batch.numpy(), axis=1)
        y_pred_batch = np.argmax(model.predict([X_batch, lengths]), axis=1)
        all_y_true_epoch.extend(y_true_batch)
        all_y_pred_epoch.extend(y_pred_batch)

    val_loss /= steps_per_epoch_val
    val_accuracy /= steps_per_epoch_val

    # Log metrics
    history['accuracy'].append(train_accuracy)
    history['val_accuracy'].append(val_accuracy)
    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)

    print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_accuracy:.4f}")
    print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}")

    # Store epoch true and predicted labels
    all_y_true.append(all_y_true_epoch)
    all_y_pred.append(all_y_pred_epoch)

    # Calculate and store confusion matrix and classification report
    conf_matrix = confusion_matrix(all_y_true_epoch, all_y_pred_epoch)
    class_report = classification_report(all_y_true_epoch, all_y_pred_epoch, target_names=['SCHZ', 'HC'],
                                         output_dict=True)
    all_conf_matrices.append(conf_matrix)
    all_class_reports.append(class_report)

    # Early stopping and model checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
        model.save('best_model.h5')
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered")
            break

# Clear session to free up resources
tf.keras.backend.clear_session()

# Calculate average metrics
average_accuracy = np.mean(history['val_accuracy'])
average_precision = np.mean([class_report['weighted avg']['precision'] for class_report in all_class_reports])
average_recall = np.mean([class_report['weighted avg']['recall'] for class_report in all_class_reports])
average_f1 = np.mean([class_report['weighted avg']['f1-score'] for class_report in all_class_reports])
average_conf_matrix = np.sum(all_conf_matrices, axis=0).astype(int)


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
average_classification_report = accumulate_classification_report(all_class_reports)

# Print the average classification report
print("Average Classification Report:")
print(f"{'Label':<15}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}{'Support':<10}")
for label, metrics in average_classification_report.items():
    if isinstance(metrics, dict):  # Ensure metrics is a dictionary
        print(
            f"{label:<15}{metrics['precision']:<10.2f}{metrics['recall']:<10.2f}{metrics['f1-score']:<10.2f}{int(metrics['support']):<10}")

# Write the average classification report to the file
with open('average_performance_metrics.txt', 'w') as f:
    f.write(f'Average Validation Accuracy across all folds: {average_accuracy:.4f}\n')
    f.write(f'Average Validation Precision across all folds: {average_precision:.4f}\n')
    f.write(f'Average Validation Recall across all folds: {average_recall:.4f}\n')
    f.write(f'Average Validation F1 Score across all folds: {average_f1:.4f}\n')
    f.write('Average Confusion Matrix:\n')
    f.write(np.array2string(average_conf_matrix))
    f.write('\nAverage Classification Report:\n')
    f.write(f"{'Label':<15}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}{'Support':<10}\n")
    for label, metrics in average_classification_report.items():
        if isinstance(metrics, dict):  # Ensure metrics is a dictionary
            f.write(
                f"{label:<15}{metrics['precision']:<10.2f}{metrics['recall']:<10.2f}{metrics['f1-score']:<10.2f}{int(metrics['support']):<10}\n")

# Plot average confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(average_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['SCHZ', 'HC'],
            yticklabels=['SCHZ', 'HC'],
            cbar_kws={'label': 'Count'}, annot_kws={"size": 14, "color": 'black'})
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Average Confusion Matrix')
plt.savefig('average_confusion_matrix.png')
plt.show()

# Calculate and print average metrics across all epochs
average_train_accuracy = np.mean(history['accuracy'])
average_val_accuracy = np.mean(history['val_accuracy'])
average_train_loss = np.mean(history['loss'])
average_val_loss = np.mean(history['val_loss'])

print(f'\nAverage Train Accuracy: {average_train_accuracy:.4f}')
print(f'Average Validation Accuracy: {average_val_accuracy:.4f}')
print(f'Average Train Loss: {average_train_loss:.4f}')
print(f'Average Validation Loss: {average_val_loss:.4f}')

with open('average_performance_metrics.txt', 'a') as f:
    f.write(f'\nAverage Train Accuracy: {average_train_accuracy:.4f}\n')
    f.write(f'Average Validation Accuracy: {average_val_accuracy:.4f}\n')
    f.write(f'Average Train Loss: {average_train_loss:.4f}\n')
    f.write(f'Average Validation Loss: {average_val_loss:.4f}\n')

# Plot average training & validation accuracy and loss values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Average Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Average Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Save average plots
plt.savefig('average_training_metrics.png')
plt.show()