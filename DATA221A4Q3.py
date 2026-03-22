# Question 3: Controlling Tree Complexity and Interpretability
# ===============================================================
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

# Load dataset
breast_cancer_data = load_breast_cancer()
feature_matrix_x = breast_cancer_data.data
target_vector_y = breast_cancer_data.target

# Split dataset
feature_train, feature_test, target_train, target_test = train_test_split(feature_matrix_x, target_vector_y, test_size=0.2, stratify = target_vector_y, random_state=42)

# Initialize and train decision tree model
decision_tree_classifier_max_depth_4 = DecisionTreeClassifier(criterion = "entropy", random_state = 42, max_depth = 4)
decision_tree_classifier_max_depth_4.fit(feature_train, target_train)

# Print test and train accuracies
train_accuracy = decision_tree_classifier_max_depth_4.score(feature_train, target_train)
test_accuracy = decision_tree_classifier_max_depth_4.score(feature_test, target_test)
print(f"Training accuracy: {train_accuracy}")
print(f"Test accuracy: {test_accuracy}")
