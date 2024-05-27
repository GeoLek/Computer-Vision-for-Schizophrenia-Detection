import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling3D
from tensorflow.data import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from scipy.ndimage import rotate, shift
import tensorflow.keras.backend as K
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from collections import defaultdict

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
    augmented = False
    if np.random.rand() > 0.5:
        volume = random_rotation(volume)
        augmented = True
    if np.random.rand() > 0.5:
        volume = random_shift(volume)
        augmented = True
    return volume, augmented

# Generator function to yield data in batches
def data_generator(data, labels, batch_size, augment=False):
    dataset_size = len(data)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    augmented_count = 0  # Counter for augmented images

    def generator():
        nonlocal augmented_count  # Use nonlocal keyword to modify the outer variable
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
                        bold_data, augmented = apply_augmentations(bold_data)
                        if augmented:
                            augmented_count += 1
                    X.append(bold_data)
                    y.append(label)

                X = np.array(X)[..., np.newaxis]  # Add channel dimension
                y = to_categorical(np.array(y), num_classes=2)  # Convert labels to categorical
                yield X, y

            if augment:
                print(f"Total augmented images this epoch: {augmented_count} out of {dataset_size} original images")  # Print total augmented images after each epoch
                augmented_count = 0  # Reset count after each epoch

    return generator

# Load data into numpy arrays
subjects, labels = [], []
data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/Applied Brain Mask & Regressed out confounds/3D Converted/'
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

# Define the simpler 3D CNN model
def create_simple_3d_cnn(input_shape, num_classes=2):
    inputs = Input(shape=input_shape)
    x = Conv3D(32, kernel_size=3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=3, strides=2, padding='same')(x)

    x = Conv3D(64, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=3, strides=2, padding='same')(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the input shape for the model
input_shape = (65, 77, 49, 1)

# Initialize k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
accuracies = []
precisions = []
recalls = []
f1s = []
conf_matrices = []
class_reports = []

# Initialize lists to store metrics across folds
all_train_accuracies = []
all_val_accuracies = []
all_train_losses = []
all_val_losses = []

for train_index, val_index in kfold.split(X_resampled):
    print(f'\nTraining fold {fold_no}...\n')
    print(f'Train indices: {train_index}\nValidation indices: {val_index}\n')
    X_train, X_val = X_resampled[train_index], X_resampled[val_index]
    y_train, y_val = y_resampled[train_index], y_resampled[val_index]

    # Convert to TensorFlow datasets using data generators
    train_gen = data_generator(X_train, y_train, batch_size, augment=True)
    val_gen = data_generator(X_val, y_val, batch_size)

    train_dataset = Dataset.from_generator(train_gen, output_types=(tf.float32, tf.float32),
                                           output_shapes=((batch_size, 65, 77, 49, 1), (batch_size, 2)))
    val_dataset = Dataset.from_generator(val_gen, output_types=(tf.float32, tf.float32),
                                         output_shapes=((batch_size, 65, 77, 49, 1), (batch_size, 2)))

    model = create_simple_3d_cnn(input_shape)
    model.summary()

    # Define early stopping and model checkpoint callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f'best_model_fold_{fold_no}.h5', monitor='val_loss', save_best_only=True, mode='min')

    # Training the model
    history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, steps_per_epoch=len(X_train) // batch_size,
                        validation_steps=len(X_val) // batch_size, callbacks=[early_stopping, checkpoint], verbose=2)

    # Append metrics to the lists
    all_train_accuracies.append(history.history['accuracy'])
    all_val_accuracies.append(history.history['val_accuracy'])
    all_train_losses.append(history.history['loss'])
    all_val_losses.append(history.history['val_loss'])

    # Save the training history as a text file
    history_file = f'training_history_fold_{fold_no}.txt'
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

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Save plots
    plt.savefig(f'training_metrics_fold_{fold_no}.png')
    plt.show()

    # Load the best model for evaluation
    best_model = tf.keras.models.load_model(f'best_model_fold_{fold_no}.h5')

    # Evaluate the model on the validation set and save performance metrics
    val_gen = data_generator(X_val, y_val, batch_size, augment=False)
    val_dataset = Dataset.from_generator(val_gen, output_types=(tf.float32, tf.float32),
                                         output_shapes=((batch_size, 65, 77, 49, 1), (batch_size, 2)))

    y_true = []
    y_pred = []

    for X_batch, y_batch in val_dataset.take(len(X_val) // batch_size):
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(best_model.predict(X_batch), axis=1))

    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrices.append(conf_matrix)
    class_report = classification_report(y_true, y_pred, target_names=['SCHZ', 'HC'])
    class_reports.append(class_report)

    # Print and save performance metrics
    print(f'\nFold {fold_no} - Validation Accuracy: {accuracy:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)

    with open(f'performance_metrics_fold_{fold_no}.txt', 'w') as f:
        f.write(f'Fold {fold_no} - Validation Accuracy: {accuracy:.4f}\n')
        f.write(f'Fold {fold_no} - Validation Precision: {precision:.4f}\n')
        f.write(f'Fold {fold_no} - Validation Recall: {recall:.4f}\n')
        f.write(f'Fold {fold_no} - Validation F1 Score: {f1:.4f}\n')
        f.write('Confusion Matrix:\n')
        f.write(np.array2string(conf_matrix))
        f.write('\nClassification Report:\n')
        f.write(class_report)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['SCHZ', 'HC'], yticklabels=['SCHZ', 'HC'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - Fold {fold_no}')
    plt.savefig(f'confusion_matrix_fold_{fold_no}.png')
    plt.show()

    fold_no += 1

# Calculate and print the average metrics across all folds
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

    for report in reports:
        for label, metrics in report.items():
            for metric, value in metrics.items():
                avg_report[label][metric] += value

    for label, metrics in avg_report.items():
        for metric in metrics:
            avg_report[label][metric] /= total_reports

    return avg_report

# Parse the classification reports into a suitable format
parsed_reports = []
for report in class_reports:
    lines = report.split('\n')
    report_dict = {}
    for line in lines[2:-3]:
        line = line.strip()
        if line:
            parts = line.split()
            class_name = parts[0]
            metrics = list(map(float, parts[1:]))
            report_dict[class_name] = {
                'precision': metrics[0],
                'recall': metrics[1],
                'f1-score': metrics[2],
                'support': metrics[3],
            }
    parsed_reports.append(report_dict)

# Calculate the average classification report
average_classification_report = accumulate_classification_report(parsed_reports)

# Print the average classification report
print("Average Classification Report:")
for label, metrics in average_classification_report.items():
    print(f"{label: <15} {metrics['precision']:.2f} {metrics['recall']:.2f} {metrics['f1-score']:.2f} {metrics['support']:.0f}")

# Write the average classification report to the file
with open('average_performance_metrics.txt', 'w') as f:
    f.write(f'Average Validation Accuracy across all folds: {average_accuracy:.4f}\n')
    f.write(f'Average Validation Precision across all folds: {average_precision:.4f}\n')
    f.write(f'Average Validation Recall across all folds: {average_recall:.4f}\n')
    f.write(f'Average Validation F1 Score across all folds: {average_f1:.4f}\n')
    f.write('Average Confusion Matrix:\n')
    f.write(np.array2string(average_conf_matrix))
    f.write('\nAverage Classification Report:\n')
    for label, metrics in average_classification_report.items():
        f.write(f"{label: <15} {metrics['precision']:.2f} {metrics['recall']:.2f} {metrics['f1-score']:.2f} {metrics['support']:.0f}\n")

# Plot average confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(average_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['SCHZ', 'HC'], yticklabels=['SCHZ', 'HC'],
            cbar_kws={'label': 'Count'}, annot_kws={"size": 14, "color": 'black'})
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Average Confusion Matrix')
plt.savefig('average_confusion_matrix.png')
plt.show()

# Calculate the average metrics
average_train_accuracy = np.mean(all_train_accuracies, axis=0)
average_val_accuracy = np.mean(all_val_accuracies, axis=0)
average_train_loss = np.mean(all_train_losses, axis=0)
average_val_loss = np.mean(all_val_losses, axis=0)

# Plot the average training & validation accuracy and loss values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(average_train_accuracy)
plt.plot(average_val_accuracy)
plt.title('Average Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(average_train_loss)
plt.plot(average_val_loss)
plt.title('Average Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Save average plots
plt.savefig('average_training_metrics.png')
plt.show()
