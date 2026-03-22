# Question 1: Dataset Exploration and Understanding
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load dataset
breast_cancer_data = load_breast_cancer()
feature_matrix_x = breast_cancer_data.data
target_vector_y = breast_cancer_data.target

# Print Shapes of X and y.
print(f"Shape of X: {feature_matrix_x.shape}")
print(f"Shape of y: {target_vector_y.shape}")

# Print number of samples belonging to each class
unique, counts = np.unique(target_vector_y, return_counts=True)
print(f"Class distribution: {dict(zip(unique, counts))}")

# ======================================================================
# Discussion:

# This dataset is imbalanced because the benign class (the one with 357 samples) has
# more samples than the malignant class (the one with 212 sample).
# Class balance matters because many classifiers can become biased toward the majority
# class, achieving high accuracy while performing poorly on the minority class. This is
# especially important in medical diagnosis like with this dataset, as misclassifying
# malignant cases can have serious consequences.