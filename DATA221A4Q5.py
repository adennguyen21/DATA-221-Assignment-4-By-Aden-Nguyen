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

# ====================================================================================