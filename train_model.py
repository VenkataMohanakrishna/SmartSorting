# Dataset directory instructions:
# Please place your dataset in the 'dataset' directory located at the same level as this file.
# The structure should be:
# dataset/
# ├── train/
# │   ├── healthy/
# │   └── rotten/
# └── test/
#     ├── healthy/
#     └── rotten/

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')

# Model parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

def build_model(num_classes):
    # Load MobileNetV2 without the top classification layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    # Freeze the base model to avoid ruining pre-trained weights
    base_model.trainable = False
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x) # prevent overfitting
    
    if num_classes == 2:
        # Binary classification (e.g. healthy vs rotten overall)
        outputs = Dense(1, activation='sigmoid')(x)
        loss_fn = 'binary_crossentropy'
    else:
        # Multi-class classification (e.g. healthy_apple, rotten_apple, etc.)
        outputs = Dense(num_classes, activation='softmax')(x)
        loss_fn = 'categorical_crossentropy'
        
    model = Model(inputs=base_model.input, outputs=outputs)
    
    model.compile(optimizer='adam', 
                  loss=loss_fn, 
                  metrics=['accuracy'])
    return model

def main():
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        print("ERROR: Dataset directories not found.")
        print(f"Please create {TRAIN_DIR} and {TEST_DIR} and place your images there.")
        print("To run a simple test without training, run 'app.py' directly.")
        # Create dummy directories to help user
        os.makedirs(os.path.join(TRAIN_DIR, 'healthy'), exist_ok=True)
        os.makedirs(os.path.join(TRAIN_DIR, 'rotten'), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, 'healthy'), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, 'rotten'), exist_ok=True)
        return

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Only rescaling for testing/validation
    test_datagen = ImageDataGenerator(rescale=1./255)

    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary' if len(os.listdir(TRAIN_DIR)) == 2 else 'categorical'
    )

    print("Loading validation data...")
    validation_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary' if len(os.listdir(TRAIN_DIR)) == 2 else 'categorical'
    )
    
    num_classes = len(train_generator.class_indices)
    print(f"Detected {num_classes} classes: {train_generator.class_indices}")

    print("Building model...")
    model = build_model(num_classes)
    
    print("Starting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )

    # Save the model
    model_path = os.path.join(BASE_DIR, 'healthy_vs_rotten.h5')
    model.save(model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == '__main__':
    main()
