# Question 4: Neural Network for Binary Classification
# ========================================================

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
import tensorflow as tf

# Load dataset
breast_cancer_data = load_breast_cancer()
feature_matrix_x = breast_cancer_data.data
target_vector_y = breast_cancer_data.target

# Split dataset
feature_train, feature_test, target_train, target_test = train_test_split(feature_matrix_x, target_vector_y, test_size=0.2, stratify = target_vector_y, random_state=42)

# Standardize input features
scaler = StandardScaler()
feature_train_x_scaled = scaler.fit_transform(feature_train)
feature_test_x_scaled = scaler.transform(feature_test)

# Train neural network model
tf.random.set_seed(1)

neural_network_model = Sequential()

neural_network_model.add(InputLayer(input_shape = (feature_train.shape[1],)))
neural_network_model.add(Dense(16)) # Hidden layer
neural_network_model.add(Dense(1, activation = "sigmoid"))

neural_network_model.compile(loss = "binary_crossentropy", metrics = ["accuracy"])

history = neural_network_model.fit(feature_train_x_scaled, target_train, epochs = 20)

# Print train and test accuracies
train_accuracy = neural_network_model.evaluate(feature_train_x_scaled, target_train)[1]
test_accuracy = neural_network_model.evaluate(feature_test_x_scaled, target_test)[1]

print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# ======================================================================
# Discussion:

# Feature scaling is necessary for neural networks because unscaled features can cause unstable
# or slow training. The weight-update process with every epoch works best when all input features
# are on a similar scale. An epoch represents one full pass through the entire training dataset
# during training, where the model updates its weights based on the observed errors.
