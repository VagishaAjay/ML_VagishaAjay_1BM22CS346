# Step 1: Import necessary libraries
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd

# Step 2: Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # feature matrix (4 features)
feature_names = iris.feature_names

# Optional: convert to DataFrame for better readability
df = pd.DataFrame(X, columns=feature_names)

# Step 3: Apply K-Means Clustering
k = 3  # we expect 3 clusters for 3 flower types
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Step 4: Add cluster labels to DataFrame
df['Cluster'] = kmeans.labels_

# Step 5: Visualize the clusters using first two features
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('K-Means Clustering on Iris Dataset')
plt.show()

