import numpy as np
from sklearn import svm
from sklearn.preprocessing import LabelEncoder

# Dataset
X = np.array([
    [1, 2],
    [2, 3],
    [3, 6],
    [6, 8],
    [7, 7]
])

y = ['A', 'A', 'B', 'B', 'B']

# Encode labels (A = 0, B = 1)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Linear SVM
model = svm.SVC(kernel='linear', C=1e5)  # high C for hard margin
model.fit(X, y_encoded)

# Get weight vector (w) and bias (b)
w = model.coef_[0]
b = model.intercept_[0]

# Hyperplane equation: w0*x + w1*y + b = 0
print(f"Hyperplane equation: {w[0]:.4f}*x + {w[1]:.4f}*y + ({b:.4f}) = 0")

# Margin = 2 / ||w||
margin = 2 / np.linalg.norm(w)
print(f"Margin width: {margin:.4f}")

# Support vectors
support_vectors = model.support_vectors_
print("\nSupport Vectors:")
for i, sv in enumerate(support_vectors):
    print(f"{i+1}) {sv}")

# Test point
test_point = np.array([[4, 5]])

# Distance from test point to hyperplane: |w·x + b| / ||w||
numerator = abs(np.dot(w, test_point[0]) + b)
denominator = np.linalg.norm(w)
distance = numerator / denominator

print(f"\nTest Point: {test_point[0]}")
print(f"Distance from hyperplane: {distance:.4f}")

# Predict label
predicted = model.predict(test_point)
predicted_label = le.inverse_transform(predicted)[0]
print(f"Predicted Label: {predicted_label}")

