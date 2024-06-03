import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.data import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from scipy.ndimage import rotate, shift
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from PIL import Image

# Custom augmentation functions
def random_rotation(image):
    angle = np.random.uniform(-10, 10)
    return rotate(image, angle, reshape=False)

def random_shift(image):
    shift_val = np.random.uniform(-5, 5, size=2)
    return shift(image, shift_val)

def apply_augmentations(image):
    augmented = False
    if np.random.rand() > 0.5:
        image = random_rotation(image)
        augmented = True
    if np.random.rand() > 0.5:
        image = random_shift(image)
        augmented = True
    return image, augmented

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
                    image_data = data[i]
                    label = labels[i]
                    if augment and label == 0:  # Only augment minority class
                        image_data, augmented = apply_augmentations(image_data)
                        if augmented:
                            augmented_count += 1
                    X.append(image_data)
                    y.append(label)

                X = np.expand_dims(np.array(X), -1)  # Add channel dimension
                y = to_categorical(np.array(y), num_classes=2)  # Convert labels to categorical
                yield X, y

            if augment:
                print(f"Total augmented images this epoch: {augmented_count} out of {dataset_size} original images")  # Print total augmented images after each epoch
                augmented_count = 0  # Reset count after each epoch

    return generator

# Load PNG images and labels into numpy arrays
subjects, labels = [], []
data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/Brain Mask_2D Converted/Downsampled'
batch_size = 8
for group in ['SCHZ', 'HC']:
    group_dir = os.path.join(data_dir, group)
    label = 0 if group == 'SCHZ' else 1
    for subject in os.listdir(group_dir):
        subject_dir = os.path.join(group_dir, subject, 'func')
        for task_file in os.listdir(subject_dir):
            if task_file.endswith('.png'):
                image_path = os.path.join(subject_dir, task_file)
                subjects.append(image_path)
                labels.append(label)

X, y = [], []
for image_path, label in zip(subjects, labels):
    image = Image.open(image_path).convert('L')
    image_data = np.array(image)
    image_data = (image_data - np.mean(image_data)) / np.std(image_data)
    X.append(image_data)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Total images loaded: {len(X)}")

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Balance the dataset by augmenting the minority class
def balance_dataset(X, y, augmentations_per_sample=2):
    X_balanced, y_balanced = [], []
    minority_class = 0  # Assuming SCHZ is the minority class

    for image, label in zip(X, y):
        X_balanced.append(image)
        y_balanced.append(label)
        if label == minority_class:
            for _ in range(augmentations_per_sample):
                augmented_image, _ = apply_augmentations(image)
                X_balanced.append(augmented_image)
                y_balanced.append(label)

    return np.array(X_balanced), np.array(y_balanced)

X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train, augmentations_per_sample=2)
print(f"Total training images after balancing: {len(X_train_balanced)}")

# Verify the balance of the training set
unique, counts = np.unique(y_train_balanced, return_counts=True)
print(f"Balanced training set class distribution: {dict(zip(unique, counts))}")

# Convert to TensorFlow datasets using data generators
train_gen = data_generator(X_train_balanced, y_train_balanced, batch_size, augment=True)
val_gen = data_generator(X_val, y_val, batch_size)

train_dataset = Dataset.from_generator(train_gen, output_types=(tf.float32, tf.float32),
                                       output_shapes=((batch_size, 48, 48, 1), (batch_size, 2)))
val_dataset = Dataset.from_generator(val_gen, output_types=(tf.float32, tf.float32),
                                     output_shapes=((batch_size, 48, 48, 1), (batch_size, 2)))

# Define the minimal 2D CNN model
def create_minimal_2d_cnn(input_shape, num_classes=2):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=2)(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the input shape for the model
input_shape = (48, 48, 1)
model = create_minimal_2d_cnn(input_shape)
model.summary()

# Verify TensorFlow is using the GPU
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
print("Using GPU:", tf.test.gpu_device_name())

# Define early stopping and model checkpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')

# Training the model
history = model.fit(train_dataset, epochs=20, validation_data=val_dataset, steps_per_epoch=len(X_train_balanced) // batch_size,
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

# Load the best model
best_model = tf.keras.models.load_model('best_model.h5')

# Evaluate the best model on the test set and save performance metrics
y_true = []
y_pred = []

for X_batch, y_batch in val_dataset.take(len(X_val) // batch_size):
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
