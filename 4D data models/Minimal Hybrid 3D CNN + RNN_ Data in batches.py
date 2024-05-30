import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, TimeDistributed, LSTM, GlobalAveragePooling3D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from collections import defaultdict
import gc

# Function to load NIfTI files
def load_nifti_file(filepath):
    img = nib.load(filepath)
    data = img.get_fdata()
    return data

# Padding or truncating the input to ensure consistent shape
def pad_or_truncate(data, target_shape):
    current_shape = data.shape
    pad_widths = [(0, max(0, t - c)) for t, c in zip(target_shape, current_shape)]
    truncated = [slice(0, min(t, c)) for t, c in zip(target_shape, current_shape)]
    data_padded = np.pad(data[tuple(truncated)], pad_width=pad_widths, mode='constant')
    return data_padded

# Data generator function to load and preprocess data in batches
def data_generator(subjects, target_shape, batch_size):
    def generator():
        np.random.shuffle(subjects)
        for start_idx in range(0, len(subjects), batch_size):
            X_batch, y_batch = [], []
            for i in range(start_idx, min(start_idx + batch_size, len(subjects))):
                bold_path, label = subjects[i]
                bold_data = load_nifti_file(bold_path)
                bold_data = (bold_data - np.mean(bold_data)) / np.std(bold_data)
                bold_data = pad_or_truncate(bold_data, target_shape)
                X_batch.append(bold_data)
                y_batch.append(label)

            X_batch = np.array(X_batch)[..., np.newaxis]  # Add channel dimension
            X_batch = np.transpose(X_batch, (0, 4, 1, 2, 3, 5))  # Reshape to (batch_size, time_steps, height, width, depth, channels)
            y_batch = to_categorical(np.array(y_batch), num_classes=2)  # Convert labels to categorical
            yield X_batch, y_batch

    return generator

# Load data paths and labels into lists
def load_data_paths(data_dir):
    subjects = []
    for group in ['SCHZ', 'HC']:
        group_dir = os.path.join(data_dir, group)
        label = 0 if group == 'SCHZ' else 1
        for subject in os.listdir(group_dir):
            subject_dir = os.path.join(group_dir, subject, 'func')
            for task_file in os.listdir(subject_dir):
                if 'preproc.nii.gz' in task_file:
                    bold_path = os.path.join(subject_dir, task_file)
                    subjects.append((bold_path, label))
    return subjects

# Define the minimal 3D CNN model
def create_minimal_3d_cnn(input_shape, num_classes=2):
    inputs = Input(shape=input_shape)
    x = Conv3D(8, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling3D(pool_size=2)(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the simplified hybrid 3D CNN + RNN model
def create_simplified_hybrid_3dcnn_rnn(input_shape, num_classes=2):
    inputs = Input(shape=input_shape)

    # Simplified 3D CNN
    x = TimeDistributed(Conv3D(8, kernel_size=3, padding='same', activation='relu'))(inputs)
    x = TimeDistributed(MaxPooling3D(pool_size=2))(x)
    x = TimeDistributed(GlobalAveragePooling3D())(x)

    # Simplified RNN Layer
    x = LSTM(16)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the input shape for the model
target_shape = (65, 77, 49, 100)
input_shape = (target_shape[3], target_shape[0], target_shape[1], target_shape[2], 1)

# Define parameters
data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/Applied Brain Mask & Regressed out confounds/'
num_batches = 500  # Number of batches to process

# Load the data paths and labels
subjects = load_data_paths(data_dir)
total_samples = len(subjects)

# Calculate the batch size
batch_size = total_samples // num_batches
if total_samples % num_batches != 0:
    batch_size += 1

accuracies = []
precisions = []
recalls = []
f1s = []
conf_matrices = []
class_reports = []

# Create data generator
train_gen = data_generator(subjects, target_shape, batch_size)
train_dataset = tf.data.Dataset.from_generator(train_gen, output_signature=(
    tf.TensorSpec(shape=(None, target_shape[3], target_shape[0], target_shape[1], target_shape[2], 1), dtype=tf.float32),
    tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
)).prefetch(tf.data.experimental.AUTOTUNE)

# Main training and evaluation loop
for batch_no, (X_batch, y_batch) in enumerate(train_dataset.take(num_batches)):
    print(f'\nProcessing batch {batch_no + 1}/{num_batches}...\n')

    # Split batch data into training (80%) and testing (20%) sets
    split_idx = int(0.8 * len(X_batch))
    X_train, X_test = X_batch[:split_idx], X_batch[split_idx:]
    y_train, y_test = y_batch[:split_idx], y_batch[split_idx:]

    model = create_simplified_hybrid_3dcnn_rnn(input_shape)
    model.summary()

    # Define early stopping and model checkpoint callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f'best_model_batch_{batch_no + 1}.h5', monitor='val_loss', save_best_only=True, mode='min')

    # Training the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=1, validation_split=0.2,  # 80% train, 20% validation
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

    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrices.append(conf_matrix)
    class_report = classification_report(y_true, y_pred, target_names=['SCHZ', 'HC'], output_dict=True)
    class_reports.append(class_report)

    # Print and save performance metrics
    print(f'\nBatch {batch_no + 1} - Test Accuracy: {accuracy:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(classification_report(y_true, y_pred, target_names=['SCHZ', 'HC']))

    with open(f'performance_metrics_batch_{batch_no + 1}.txt', 'w') as f:
        f.write(f'Batch {batch_no + 1} - Test Accuracy: {accuracy:.4f}\n')
        f.write(f'Batch {batch_no + 1} - Test Precision: {precision:.4f}\n')
        f.write(f'Batch {batch_no + 1} - Test Recall: {recall:.4f}\n')
        f.write(f'Batch {batch_no + 1} - Test F1 Score: {f1:.4f}\n')
        f.write('Confusion Matrix:\n')
        f.write(np.array2string(conf_matrix))
        f.write('\nClassification Report:\n')
        f.write(classification_report(y_true, y_pred, target_names=['SCHZ', 'HC']))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['SCHZ', 'HC'], yticklabels=['SCHZ', 'HC'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - Batch {batch_no + 1}')
    plt.savefig(f'confusion_matrix_batch_{batch_no + 1}.png')
    plt.show()

    # Clear session and delete model to free up memory
    tf.keras.backend.clear_session()
    del model
    del best_model
    gc.collect()

# Calculate and print the average metrics across all batches
average_accuracy = np.mean(accuracies)
average_precision = np.mean(precisions)
average_recall = np.mean(recalls)
average_f1 = np.mean(f1s)

# Calculate the average confusion matrix
average_conf_matrix = np.sum(conf_matrices, axis=0).astype(int)

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
    f.write(f'Average Test Accuracy across all batches: {average_accuracy:.4f}\n')
    f.write(f'Average Test Precision across all batches: {average_precision:.4f}\n')
    f.write(f'Average Test Recall across all batches: {average_recall:.4f}\n')
    f.write(f'Average Test F1 Score across all batches: {average_f1:.4f}\n')
    f.write('Average Confusion Matrix:\n')
    f.write(np.array2string(average_conf_matrix))
    f.write('\nAverage Classification Report:\n')
    f.write(f"{'Label':<15}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}{'Support':<10}\n")
    for label, metrics in average_classification_report.items():
        if isinstance(metrics, dict):  # Ensure metrics is a dictionary
            f.write(f"{label:<15}{metrics['precision']:<10.2f}{metrics['recall']:<10.2f}{metrics['f1-score']:<10.2f}{int(metrics['support']):<10}\n")

# Plot average confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(average_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['SCHZ', 'HC'], yticklabels=['SCHZ', 'HC'],
            cbar_kws={'label': 'Count'}, annot_kws={"size": 14, "color": 'black'})
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Average Confusion Matrix')
plt.savefig('average_confusion_matrix.png')
plt.show()
