# Question 6: Convolutional Neural Network with Built-in Dataset
# ===============================================================

from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf
from tensorflow.keras import layers, models

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

# Print test accuracy
test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test, verbose = 0)
print(f"Test accuracy: {test_accuracy}")

# =================================================================================
# Discussion:

# CNNs are preferred over fully connected networks for image data because they take advantage
# of spatial patterns (nearby pixels that are related), and they use far fewer parameters.
# The convolution layer learns local features such as edges, textures, and simple shapes, which
# are combined in deeper layers to recognize more complex patterns in clothing images.
