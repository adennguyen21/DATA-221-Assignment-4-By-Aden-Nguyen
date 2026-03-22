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

print("======================================")
# Print top five most important features
importances = decision_tree_classifier_max_depth_4.feature_importances_

# Gets the top five most important indices
indices = np.argsort(importances)[::-1][:5]

print("Top 5 most important features:")
for index in indices:
    print(f"{breast_cancer_data.feature_names[index]}: {importances[index]}")

# ===========================================================
# Discussion:

# Controlling the complexity of the tree (like limiting depth) helps prevent overfitting, since
# it reduces model complexity by forcing the model to learn broader, more general patterns,
# instead of just memorizing the training data.
# Feature importance values improve interpretability because they show which features the model
# relies on the most when making decisions, allowing us to understand the model's behavior.