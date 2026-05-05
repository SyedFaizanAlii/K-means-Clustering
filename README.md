# 🔵 K-Means Clustering — A Complete Learning Guide

Master K-Means Clustering — one of the most powerful and widely-used unsupervised machine learning algorithms. This repository covers theory, mathematics, implementation, and real-world applications.

---

## 📌 Core Concepts
* **Unsupervised Learning:** Clustering groups similar data points without pre-labeled categories.
* **Centroids:** The algorithm partitions data into **K** non-overlapping clusters by minimizing the distance between points and their cluster's mean center (centroid).
* **WCSS (Inertia):** K-Means optimizes the Within-Cluster Sum of Squares to ensure cluster tightness.

## 🛠️ Key Strategies for Optimization
* **Choosing K:** Use the **Elbow Method** (plotting WCSS vs K) and **Silhouette Scores** (-1 to +1) to find the optimal number of clusters.
* **Smarter Initialization:** Utilize **K-Means++** to spread out initial centroids intelligently, avoiding the pitfalls of random placement.
* **Efficiency:** Use **Mini-Batch K-Means** for large datasets to speed up computation with minimal accuracy loss.

## 🚀 Quick Implementation (Scikit-Learn)
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Always scale features for distance-based algorithms[cite: 6]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit K-Means with smart initialization[cite: 6]
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)
