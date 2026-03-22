# Question 1: Dataset Exploration and Understanding

from sklearn.datasets import load_breast_cancer

# Load dataset
breast_cancer_data = load_breast_cancer()
feature_matrix_x = breast_cancer_data.data
target_vector_y = breast_cancer_data.target

# Print Shapes of X and y.
print(f"Shape of X: {feature_matrix_x.shape}")
print(f"Shape of y: {target_vector_y.shape}")
