import kagglehub
ciplab_real_and_fake_face_detection_path = kagglehub.dataset_download('ciplab/real-and-fake-face-detection')

print('Data source import complete.')

import numpy as no
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm.notebook import tqdm_notebook as tqdm
import os

"""import the required library to perform the task"""

print(os.listdir("../input"))

real = "../input/real-and-fake-face-detection/real_and_fake_face/training_real/"
fake = "../input/real-and-fake-face-detection/real_and_fake_face/training_fake/"

real_path = os.listdir(real)
fake_path = os.listdir(fake)
print("successfully exceuted!")

def load_img(path):
    image = cv2.imread(path)
    image = cv2.resize(image,(224, 224))
    return image[...,::-1]

"""to visualize the image in real face"""

fig = plt.figure(figsize=(10, 10))

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(load_img(real + real_path[i]), cmap='gray')
    plt.suptitle("Real faces",fontsize=20)
    plt.axis('off')

plt.show()

"""to visualize the fake image"""

fig = plt.figure(figsize=(10, 10))

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(load_img(fake + fake_path[i]), cmap='gray')
    plt.suptitle("fake faces",fontsize=20)
    plt.axis('off')

plt.show()

"""Data augmentation"""

dataset="/kaggle/input/real-and-fake-face-detection/real_and_fake_face"
# "/kaggle/input/real-and-fake-face-detection/real_and_fake_face"

import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_with_aug = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=False,
    rescale=1./255,
    validation_split=0.2)

train = data_with_aug.flow_from_directory(dataset,
                                           class_mode="binary",
                                           target_size=(96, 96),
                                            batch_size=32,
                                           subset="training")

val = data_with_aug.flow_from_directory(dataset,
                                        class_mode="binary",
                                        target_size=(96, 96),
                                        batch_size=32,
                                        subset="validation")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense,BatchNormalization, Flatten, MaxPool2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from keras.layers import Conv2D, Reshape
from keras.utils import Sequence
from keras.backend import epsilon
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

"""apply the transfer learning with pretrained model VGGNET"""

from keras.applications.vgg16 import VGG16

"""create a object of this model"""

# Load base model without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(96, 96, 3))

base_model.summary()

"""perform the feature extraction => freeeze all layer"""

base_model.trainable=False

base_model.summary()

"""Model building"""

model=Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# Compile
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

"""train the model

to reduce the overfitting we are using the Earlystopping concept
"""

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True,verbose=1)

import warnings
warnings.filterwarnings("ignore")

#train model
history1= model.fit(train, epochs=25, validation_data=val, callbacks=[early_stop])

"""fine tuning"""

# Load base model without top layers
base_model1 = VGG16(weights='imagenet', include_top=False, input_shape=(96, 96, 3))

for layer in base_model1.layers[-5:]:  # unfreeze last 30 layers
    layer.trainable = True

base_model1.summary()

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

# Assuming you already have a pretrained base model like:
# base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

model = Sequential()
model.add(base_model1)  # ✅ fixed typo
model.add(GlobalAveragePooling2D())

# Dense layers with batch normalization and dropout for stability
model.add(Dense(256, kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

# Output layer for binary classification (real vs fake)
model.add(Dense(1, activation='sigmoid'))

model.summary()

# Compile
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

#train model
history2= model.fit(train, epochs=25, validation_data=val, callbacks=[early_stop])

val_accuracies = history2.history['val_accuracy']

# Get the highest validation accuracy
best_val_acc_vgg= max(val_accuracies)

print(f"Best Validation Accuracy: {best_val_acc_vgg:.4f}")

test_loss, test_accuracy = model.evaluate(val)
print(f"Test Loss: {test_loss}")  # Printing the loss on test data
print(f"Test Accuracy: {test_accuracy}")

# Plot training accuracy
plt.plot(history2.history['accuracy'], label='Training Accuracy')
plt.plot(history2.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()  # Display the accuracy plot

# Plot training and validation loss over the epochs
plt.plot(history2.history['loss'], label='Training Loss')
plt.plot(history2.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""use different model MobileNetV2"""

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.regularizers import l1_l2

# Load base model without top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(96, 96, 3))

# Freeze all layers except the last 10
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

# Create final model
model_resnet50 = Model(inputs=base_model.input, outputs=output)

# Compile
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model_resnet50.compile(optimizer=optimizer,
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

model_resnet50.summary()

history4= model_resnet50.fit(train, epochs=25, validation_data=val, callbacks=[early_stop])

# Get all validation accuracies per epoch
val_accuracies = history4.history['val_accuracy']

# Get the highest validation accuracy
best_val_acc_resnet = max(val_accuracies)

print(f"Best Validation Accuracy: {best_val_acc_resnet:.4f}")

# Evaluate the MobileNetV2 model on the test (validation) data to check its performance
test_loss, test_accuracy = model_resnet50.evaluate(val)
print(f"Test Loss: {test_loss}")  # Printing the loss on test data
print(f"Test Accuracy: {test_accuracy}")  # Printing the accuracy on test data

# Plot training accuracy
plt.plot(history4.history['accuracy'], label='Training Accuracy')
plt.plot(history4.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()  # Display the accuracy plot

# Plot training and validation loss over the epochs
plt.plot(history4.history['loss'], label='Training Loss')
plt.plot(history4.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()  # Display the loss plot

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np

# List of model names (must match the models you trained)
model_names = ['VGG16', 'ResNet50']

# Validation accuracies (replace these with your actual model results)
val_accuracies = [0.5637, 0.5882]

# Find the index and name of the best performing model
best_model_idx = np.argmax(val_accuracies)
best_model_name = model_names[best_model_idx]

# Create a bar chart to compare validation accuracies
plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, val_accuracies, color='skyblue', edgecolor='black')

# Add percentage labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.002, f'{yval:.2%}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# Set chart title and axis labels
plt.title('Validation Accuracy Comparison of CNN Models', fontsize=16, fontweight='bold')
plt.xlabel('Model', fontsize=14)
plt.ylabel('Validation Accuracy', fontsize=14)

# Set y-axis range with some margin
plt.ylim(0, max(val_accuracies) + 0.05)

# Add gridlines for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Highlight the best model bar in green
bars[best_model_idx].set_color('limegreen')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# Print the best model and its accuracy
print(f"🏆 Best Model: {best_model_name} with Validation Accuracy = {max(val_accuracies):.2%}")

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def predict_real_or_fake(model, img_path):
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(96, 96))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # normalize

        # Predict
        prediction = model.predict(img_array)[0][0]

        # Map prediction to label
        if prediction > 0.5:
            label = "Real Face"
            confidence = prediction * 100
        else:
            label = "Fake Face"
            confidence = (1 - prediction) * 100

        print(f"🧠 Prediction: {label} ({confidence:.2f}% confidence)")
        return label, confidence

    except Exception as e:
        print(f"Error predicting image: {e}")
        return None, None

