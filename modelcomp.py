import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd  # For table generation

# Define function to build custom CNN model
def build_custom_cnn(input_shape=(150, 150, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    return model

# Load and preprocess data
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    r'C:\Users\nithi\OneDrive\Desktop\final year project\pneumonia-detection-flask\static\uploads\train',  # Replace with your dataset path
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    r'C:\Users\nithi\OneDrive\Desktop\final year project\pneumonia-detection-flask\static\uploads\test',  # Replace with your dataset path
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Compile and train model
def train_model(model, train_data, val_data, epochs=10):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_data, validation_data=val_data, epochs=epochs)
    return history

# Compare models and store metrics in a table
def compare_models_with_table(train_data, val_data, epochs=10):
    # Custom CNN
    print("Training Custom_CNN...")
    custom_cnn = build_custom_cnn()
    history_custom_cnn = train_model(custom_cnn, train_data, val_data, epochs)
    
    # Pre-trained models
    models = {
        "VGG16": tf.keras.applications.VGG16(weights=None, input_shape=(150, 150, 3), classes=1, classifier_activation='sigmoid'),
        "ResNet50": tf.keras.applications.ResNet50(weights=None, input_shape=(150, 150, 3), classes=1, classifier_activation='sigmoid'),
        "MobileNet": tf.keras.applications.MobileNet(weights=None, input_shape=(150, 150, 3), classes=1),
        "EfficientNetB0": tf.keras.applications.EfficientNetB0(weights=None, input_shape=(150, 150, 3), classes=1, classifier_activation='sigmoid')
    }
    
    histories = {"Custom_CNN": history_custom_cnn}
    metrics = []  # Store accuracy and loss details
    
    # Collect metrics for custom CNN
    metrics.append({
        "Model": "Custom_CNN",
        "Train Accuracy": history_custom_cnn.history['accuracy'][-1],
        "Validation Accuracy": history_custom_cnn.history['val_accuracy'][-1],
        "Train Loss": history_custom_cnn.history['loss'][-1],
        "Validation Loss": history_custom_cnn.history['val_loss'][-1]
    })
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        history = model.fit(train_data, validation_data=val_data, epochs=epochs)
        histories[name] = history
        # Collect metrics for pre-trained models
        metrics.append({
            "Model": name,
            "Train Accuracy": history.history['accuracy'][-1],
            "Validation Accuracy": history.history['val_accuracy'][-1],
            "Train Loss": history.history['loss'][-1],
            "Validation Loss": history.history['val_loss'][-1]
        })
    
    return histories, pd.DataFrame(metrics)

# Plot results and display table
def plot_results_and_table(histories, metrics_df):
    # Plot accuracy
    plt.figure(figsize=(12, 8))
    for name, history in histories.items():
        plt.plot(history.history['accuracy'], label=f'{name} Train Accuracy')
        plt.plot(history.history['val_accuracy'], label=f'{name} Validation Accuracy')
    plt.title('Model Comparison: Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot loss
    plt.figure(figsize=(12, 8))
    for name, history in histories.items():
        plt.plot(history.history['loss'], label=f'{name} Train Loss')
        plt.plot(history.history['val_loss'], label=f'{name} Validation Loss')
    plt.title('Model Comparison: Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Display table
    print("Model Performance Summary:")
    print(metrics_df)

# Main execution
if __name__ == "__main__":
    histories, metrics_df = compare_models_with_table(train_data, val_data, epochs=10)
    plot_results_and_table(histories, metrics_df)
