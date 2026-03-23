# Question 5: Model Evaluation and Comparison
# ============================================

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
import tensorflow as tf
import numpy as np
from sklearn.tree import DecisionTreeClassifier

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

# ===================================================================

# Initialize and train decision tree model
decision_tree_classifier_max_depth_4 = DecisionTreeClassifier(criterion = "entropy", random_state = 42, max_depth = 4)
decision_tree_classifier_max_depth_4.fit(feature_train, target_train)

# Print confusion matrix for decision tree model
y_predicted_tree = decision_tree_classifier_max_depth_4.predict(feature_test)
confusion_matrix_tree = confusion_matrix(target_test, y_predicted_tree)
print(f"Decision Tree Confusion Matrix:\n {confusion_matrix_tree}")

# =========================================================================================

# Train neural network model
tf.random.set_seed(1)

neural_network_model = Sequential()

neural_network_model.add(InputLayer(input_shape = (feature_train.shape[1],)))
neural_network_model.add(Dense(16)) # Hidden layer
neural_network_model.add(Dense(1, activation = "sigmoid"))

neural_network_model.compile(loss = "binary_crossentropy", metrics = ["accuracy"])

history = neural_network_model.fit(feature_train_x_scaled, target_train, epochs = 20)

# Print confusion matrix for neural network
y_predicted_neural_network = (neural_network_model.predict(feature_test_x_scaled) > 0.5).astype(int)
confusion_matrix_neural_network = confusion_matrix(target_test, y_predicted_neural_network)
print(f"Neural Network Confusion Matrix:\n {confusion_matrix_neural_network}")