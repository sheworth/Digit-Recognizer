# Digit-Recognizer
# Project Overview:

 This project leverages the MNIST dataset to develop machine learning models for recognizing handwritten digits. We explore two approaches: a fully connected neural network (FNN) and a Convolutional Neural Network (CNN), comparing their performance in terms of accuracy and loss metrics. The models are trained, evaluated, and validated on a standardized test set. Finally, we use a confusion matrix to assess model predictions across all classes.
 
 # Table of Contents
 
Project Overview

Data Overview

Exploratory Data Analysis

Modeling

Evaluation

Future Work

Conclusion

License

# Required libraries:

numpy

pandas

matplotlib

seaborn

tensorflow

scikit-learn

# Data Overview

We use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits. Each image is represented as a 28x28 grayscale pixel array.

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()  

print(f"X_train shape: {X_train.shape}")  # (60000, 28, 28)  
print(f"Y_train shape: {Y_train.shape}")  # (60000,)  
print(f"X_test shape: {X_test.shape}")    # (10000, 28, 28)  
print(f"Y_test shape: {Y_test.shape}")    # (10000,)  
Feature: Grayscale pixel values ranging from 0 to 255.
Target: Digit labels (0-9).

# Exploratory Data Analysis

# Visualizing Sample Images

import matplotlib.pyplot as plt  

plt.imshow(X_train[700], cmap='cividis_r')  

plt.title(f"Label: {Y_train[700]}")  

plt.show()  

# Displaying 10 Samples per Class

class_images = {i: [] for i in range(10)}  

for i, (image, label) in enumerate(zip(X_train, Y_train)):  
    if len(class_images[label]) < 10:  
        class_images[label].append(image)  
    if all(len(images) == 10 for images in class_images.values()):  
        break  

fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(12, 12))  
plt.subplots_adjust(hspace=0.5)  

for i in range(10):  
    for j in range(10):  
        axes[i, j].imshow(class_images[i][j], cmap='cividis_r')  
        axes[i, j].axis('off')  
        if j == 0:  
            axes[i, j].set_ylabel(f'Digit {i}', rotation=0, labelpad=30, fontsize=10)  

plt.show()  

Modeling

1. # Fully Connected Neural Network (FNN)
   
from tensorflow import keras  

model = keras.Sequential([  
    keras.layers.Input(shape=(28, 28)),  
    keras.layers.Flatten(),  
    keras.layers.Dense(50, activation='relu'),  
    keras.layers.Dense(50, activation='relu'),  
    keras.layers.Dense(10, activation='softmax')  
])  

model.compile(optimizer='adam',  
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy'])  

model.fit(X_train, Y_train, epochs=10)  

2. # Convolutional Neural Network (CNN)
   
model = keras.Sequential([  
    keras.layers.Input(shape=(28, 28, 1)),  
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),  
    keras.layers.MaxPooling2D(pool_size=(2, 2)),  
    keras.layers.Flatten(),  
    keras.layers.Dense(128, activation='relu'),  
    keras.layers.Dense(10, activation='softmax')  
])  

model.compile(optimizer='adam',  
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy'])  

model.fit(X_train, Y_train, epochs=10)  
Evaluation
Model Performance on Test Set

import pandas as pd  

loss, accuracy = model.evaluate(X_test, Y_test)  

results = pd.DataFrame({  
    'Metric': ['Loss', 'Accuracy'],  
    'Value': [loss, accuracy]  
})  

print(results)  
Sample Output:
     Metric     Value  
0      Loss  0.057234  
1  Accuracy  0.986600  

# Confusion Matrix

import numpy as np  

import seaborn as sns 

from sklearn.metrics import confusion_matrix  

Y_pred = model.predict(X_test)  

Y_pred_classes = np.argmax(Y_pred, axis=1)  

cm = confusion_matrix(Y_test, Y_pred_classes)  

plt.figure(figsize=(10, 7))  
sns.heatmap(cm, annot=True, fmt='d', cmap='cividis_r')  
plt.xlabel('Predicted')  
plt.ylabel('Actual')  
plt.show()  

# Future Work

Hyperparameter Tuning: Using tools like KerasTuner to optimize learning rates, batch size, and architecture depth.

Data Augmentation: Implement transformations such as rotation, scaling, and shifting to improve generalization.

Deploying the Model: Deploy the trained model using Flask, FastAPI, or Streamlit.

Model Interpretability: Integrate SHAP or LIME to interpret predictions.

# Conclusion

This project demonstrates two models for digit recognition using the MNIST dataset. While the fully connected network performs adequately, the CNN model achieves a higher accuracy due to its ability to capture spatial dependencies in the image data. The CNN achieved a test accuracy of 98.66% and shows potential for further improvement with techniques like hyperparameter tuning and data augmentation.

License
This project is licensed under the MIT License - see the LICENSE file for details.

