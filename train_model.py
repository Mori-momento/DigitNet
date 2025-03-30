import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

print("TensorFlow Version:", tf.__version__)

# --- Configuration ---
MODEL_SAVE_PATH = 'mnist_cnn_best.h5' # Save the best model found
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)
BATCH_SIZE = 128
EPOCHS = 20 # Increase epochs, EarlyStopping will prevent overfitting
VALIDATION_SPLIT = 0.2 # Use a portion of training data for validation

# --- 1. Load and Preprocess Data ---
print("Loading MNIST data...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape data to add channel dimension (samples, height, width, channels)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
print("y_train shape:", y_train.shape) # Should be (60000,)

# Note: We are using sparse_categorical_crossentropy, so y_train/y_test remain as integers.
# If using categorical_crossentropy, you would convert y_train/y_test using:
# y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
# y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

# --- 2. Data Augmentation ---
# Create a data generator for training data with augmentation
train_datagen = ImageDataGenerator(
    rotation_range=10,       # Randomly rotate images by 10 degrees
    width_shift_range=0.1,   # Randomly shift images horizontally by 10%
    height_shift_range=0.1,  # Randomly shift images vertically by 10%
    zoom_range=0.1,          # Randomly zoom images by 10%
    shear_range=0.1          # Shear intensity
)
# Fit the generator on the training data (calculates internal stats if needed)
# For MNIST, this step might not be strictly necessary for these transforms
train_datagen.fit(x_train)

# Create the training data iterator
# Note: We use VALIDATION_SPLIT in model.fit, so we provide the full x_train here
# If using validation_data= in fit, you'd split manually first.
# train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

# --- 3. Build the CNN Model ---
print("Defining improved CNN model...")
model = keras.Sequential(
    [
        keras.Input(shape=INPUT_SHAPE),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5), # Dropout for regularization
        layers.Dense(128, activation="relu"), # Slightly larger Dense layer
        layers.Dropout(0.3), # More dropout
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model.summary()

# --- 4. Compile the Model ---
model.compile(loss="sparse_categorical_crossentropy", # Use sparse version as y_train is integers
              optimizer="adam",
              metrics=["accuracy"])

# --- 5. Define Callbacks ---
# Save the best model based on validation loss
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH,
                             monitor='val_accuracy', # Monitor validation accuracy
                             verbose=1,
                             save_best_only=True, # Only save if improvement
                             mode='max') # maximize accuracy

# Stop training early if validation loss stops improving
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=3, # Stop after 3 epochs of no improvement
                               verbose=1,
                               mode='min', # minimize loss
                               restore_best_weights=True) # Restore weights from the best epoch

callbacks_list = [checkpoint, early_stopping]

# --- 6. Train the Model ---
print(f"\nTraining model for up to {EPOCHS} epochs...")

# Train using the data generator for augmentation
# model.fit will automatically use a portion of x_train for validation due to validation_split
history = model.fit(train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                    # Calculate steps per epoch for generator
                    steps_per_epoch=int(len(x_train) * (1-VALIDATION_SPLIT)) // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(x_test, y_test), # Use the actual test set for validation here
                    # Alternatively, use validation_split with model.fit(x_train, y_train...) if not using generator
                    # validation_split=VALIDATION_SPLIT,
                    callbacks=callbacks_list,
                    verbose=1) # Set verbose=1 or 2 to see progress


# --- 7. Evaluate the Best Model ---
print("\nEvaluating the best model saved...")
# Load the best saved model (optional, as EarlyStopping might restore best weights)
# model = keras.models.load_model(MODEL_SAVE_PATH)

# Evaluate on the test set
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {score[0]:.4f}")
print(f"Test Accuracy: {score[1]:.4f}")

print(f"\nBest model saved to {MODEL_SAVE_PATH}")