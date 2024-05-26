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
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# Data augmentation using ImageDataGenerator
augmentation = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'
)

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

            print(f"Total augmented images this epoch: {augmented_count} out of {dataset_size} original images")  # Print total augmented images after each epoch
            augmented_count = 0  # Reset count after each epoch

    return generator

# Load data into numpy arrays
subjects, labels = [], []
data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/Applied Brain Mask & Regressed out confounds/3D Converted/'
batch_size = 16
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

# Initialize k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
accuracies = []

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

    model = create_3d_resnet(input_shape)
    model.summary()

    # Define early stopping and model checkpoint callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f'best_model_fold_{fold_no}.h5', monitor='val_loss', save_best_only=True, mode='min')

    # Training the model
    history = model.fit(train_dataset, epochs=5, validation_data=val_dataset, steps_per_epoch=len(X_train) // batch_size,
                        validation_steps=len(X_val) // batch_size, callbacks=[early_stopping, checkpoint], verbose=2)

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

    # Plot training & validation loss values
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
    val_gen = data_generator(X_val, y_val, batch_size)
    val_dataset = Dataset.from_generator(val_gen, output_types=(tf.float32, tf.float32),
                                         output_shapes=((batch_size, 65, 77, 49, 1), (batch_size, 2)))

    y_true = []
    y_pred = []

    for X_batch, y_batch in val_dataset.take(len(X_val) // batch_size):
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(best_model.predict(X_batch), axis=1))

    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    accuracies.append(accuracy)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=['SCHZ', 'HC'])

    # Print and save performance metrics
    print(f'\nFold {fold_no} - Validation Accuracy: {accuracy:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)

    with open(f'performance_metrics_fold_{fold_no}.txt', 'w') as f:
        f.write(f'Fold {fold_no} - Validation Accuracy: {accuracy:.4f}\n')
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

# Print the average accuracy across all folds
average_accuracy = np.mean(accuracies)
print(f'Average Validation Accuracy across all folds: {average_accuracy:.4f}')
