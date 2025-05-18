import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Step 1: Load and Standardize the dataset
iris = load_iris()
X = iris.data
X_std = StandardScaler().fit_transform(X)

# Step 2: Compute Covariance Matrix
cov_matrix = np.cov(X_std.T)

# Step 3: Compute Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 4: Sort eigenvectors by descending eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, sorted_indices]
eigenvalues = eigenvalues[sorted_indices]

# Step 5: Select top 2 eigenvectors and project data
n_components = 2
top_eigenvectors = eigenvectors[:, :n_components]
X_reduced = X_std @ top_eigenvectors

# Output the PCA components (2D transformed data)
print("PCA Components (first 5 rows):\n", X_reduced[:5])
