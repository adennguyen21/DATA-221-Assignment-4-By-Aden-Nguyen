# Question 2: Decision Tree Model Using Entropy
# =================================================

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

# Load dataset
breast_cancer_data = load_breast_cancer()
feature_matrix_x = breast_cancer_data.data
target_vector_y = breast_cancer_data.target

# Split dataset
feature_train, feature_test, target_train, target_test = train_test_split(feature_matrix_x, target_vector_y, test_size=0.2, stratify = target_vector_y, random_state=42)

# Initialize and train decision tree model
decision_tree_classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 42)
decision_tree_classifier.fit(feature_train, target_train)

# Print test and train accuracies
train_accuracy = decision_tree_classifier.score(feature_train, target_train)
test_accuracy = decision_tree_classifier.score(feature_test, target_test)
print(f"Training accuracy: {train_accuracy}")
print(f"Test accuracy: {test_accuracy}")




