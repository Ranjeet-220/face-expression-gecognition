import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
from src.config import INPUT_SHAPE, BATCH_SIZE, EPOCHS, NUM_CLASSES, DATA_DIR, MODELS_DIR, PROJECT_ROOT
from src.model import build_model
import argparse

def train(data_dir=DATA_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Trains the emotion recognition model.
    Assumes data_dir contains 'train' and 'validation' subdirectories.
    """
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'validation')

    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found at {train_dir}")
        return

    # Data Augmentation (Advanced)
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        channel_shift_range=20.0,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    print(f"Loading data from {train_dir} and {val_dir}...")
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=INPUT_SHAPE[:2],
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=INPUT_SHAPE[:2],
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Build Model
    model = build_model()
    
    # Checkpoints
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    checkpoint_path = os.path.join(MODELS_DIR, "emotion_model.h5")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Callbacks
    tensorboard_log_dir = os.path.join(PROJECT_ROOT, "logs")
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    tensorboard = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)

    # Train
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop, reduce_lr, tensorboard]
    )
    
    print("Training finished.")
    
    # Plotting
    plot_path = os.path.join(PROJECT_ROOT, "training_plot.png")
    plot_history(history, plot_path)
    print(f"Training plot saved to {plot_path}")

def plot_history(history, save_path):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Path to data directory containing train/validation folders")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    
    train(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
