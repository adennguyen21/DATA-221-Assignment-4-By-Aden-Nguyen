# Question 7: CNN Error Analysis and Misclassification Study
# ============================================================

from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

print("TF version:", tf.__version__)

# Split dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize and Reshape
X_train = X_train.astype("float32")/255.0
X_test = X_test.astype("float32")/255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Build CNN
cnn_model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, padding = "same", activation = "relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(10, activation = "softmax")
])

cnn_model.compile(optimizer = "adam",
                  loss = "sparse_categorical_crossentropy",
                  metrics = ["accuracy"])

cnn_model.fit(X_train, y_train,
              epochs = 15,
              validation_split = 0.1,
              batch_size = 64)

# Predictions
y_predicted = cnn_model(X_test)
y_predicted_labels = np.argmax(y_predicted, axis = 1)

# Print confusion matrix of CNN model
confusion_matrix_cnn = confusion_matrix(y_test, y_predicted_labels)
print(f"Confusion Matrix:\n {confusion_matrix_cnn}")

# Find three misclassified indices
mis_index = (y_predicted_labels != y_test)
misclassified_images = np.where(mis_index)[0][:3]

# Display misclassified images
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

for index in misclassified_images:
    plt.imshow(X_test[index].reshape(28, 28), cmap = "gray")
    plt.title(f"True label: {y_test[index]} ({class_names[y_test[index]]}), Predicted label: {y_predicted_labels[index]} ({class_names[y_predicted_labels[index]]})")
    plt.show()
