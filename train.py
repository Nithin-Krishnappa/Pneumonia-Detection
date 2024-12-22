import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint

# Set the path to your dataset
train_dir = r"C:\Users\nithi\OneDrive\Desktop\final year project\pneumonia-detection-flask\static\uploads\train"
validation_dir = r"C:\Users\nithi\OneDrive\Desktop\final year project\pneumonia-detection-flask\static\uploads\test"

# Image Data Augmentation and Normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Image size
    batch_size=32,
    class_mode='binary'  # Binary classification: Pneumonia or Not Pneumonia
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Build CNN Model (Fine-tuning a pre-trained model like VGG16 or ResNet50)
base_model = tf.keras.applications.VGG16(input_shape=(150, 150, 3),
                                          include_top=False,
                                          weights='imagenet')

# Freeze base model layers
base_model.trainable = False

# Create new model on top
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile the model
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Save the best model based on validation accuracy
checkpoint = ModelCheckpoint("pneumonia_model.keras", 
                             save_best_only=True, 
                             monitor='val_accuracy', 
                             mode='max')

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[checkpoint]
)

# Save the model to disk
model.save("pneumonia_model.keras")
