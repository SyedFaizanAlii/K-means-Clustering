# 🔵 K-Means Clustering — A Complete Learning Guide

> **Understand and master K-Means Clustering — one of the most powerful and widely-used unsupervised machine learning algorithms.**  
> This repository is a comprehensive learning resource covering the theory, mathematics, implementation, and real-world applications of K-Means and related clustering algorithms.

---

## 📌 Table of Contents

1. [What is Clustering?](#1-what-is-clustering)
2. [What is K-Means Clustering?](#2-what-is-k-means-clustering)
3. [How K-Means Works — Step by Step](#3-how-k-means-works--step-by-step)
4. [The Mathematics Behind K-Means](#4-the-mathematics-behind-k-means)
5. [Choosing K — The Elbow Method & More](#5-choosing-k--the-elbow-method--more)
6. [K-Means Variants](#6-k-means-variants)
7. [Implementation from Scratch](#7-implementation-from-scratch)
8. [K-Means with Scikit-Learn](#8-k-means-with-scikit-learn)
9. [Evaluating Clustering Quality](#9-evaluating-clustering-quality)
10. [Visualizing Clusters](#10-visualizing-clusters)
11. [Real-World Applications](#11-real-world-applications)
12. [Related Clustering Algorithms](#12-related-clustering-algorithms)
13. [Full Project Examples](#13-full-project-examples)
14. [Advantages & Limitations](#14-advantages--limitations)
15. [Tips & Best Practices](#15-tips--best-practices)
16. [Common Mistakes to Avoid](#16-common-mistakes-to-avoid)
17. [Resources & References](#17-resources--references)

---

## 1. What is Clustering?

**Clustering** is an **unsupervised machine learning** technique that groups similar data points together — **without any pre-labeled categories**.

> Unlike supervised learning (where you have labels like "spam / not spam"), clustering discovers hidden structure and natural groupings in data on its own.

### Where is clustering used?
| Domain | Use Case |
|---|---|
| 🛒 E-commerce | Customer segmentation by behavior |
| 🏥 Healthcare | Grouping patients by symptoms/genetics |
| 📰 NLP | Topic modeling, document grouping |
| 🖼️ Image Processing | Image compression, pixel segmentation |
| 🔒 Cybersecurity | Anomaly/intrusion detection |
| 🎵 Music / Netflix | Recommendation systems |
| 🌍 Geography | City planning, traffic analysis |

---

## 2. What is K-Means Clustering?

**K-Means** is the most popular clustering algorithm. It partitions `n` data points into **K non-overlapping clusters** by minimizing the distance between each point and the **centroid** (mean center) of its assigned cluster.

### Key Terms:
| Term | Meaning |
|---|---|
| **K** | Number of clusters (you decide this) |
| **Centroid** | The mean center point of a cluster |
| **Inertia / WCSS** | Within-Cluster Sum of Squares — measures cluster tightness |
| **Assignment Step** | Assign each point to its nearest centroid |
| **Update Step** | Recalculate centroids as mean of assigned points |
| **Convergence** | When assignments no longer change |

---

## 3. How K-Means Works — Step by Step

```
ALGORITHM: K-Means

INPUT : Dataset X with n points, K (number of clusters)
OUTPUT: K cluster assignments + K centroids

─────────────────────────────────────────────────
STEP 1: INITIALIZE
  → Randomly select K data points as initial centroids
  → (or use K-Means++ for smarter initialization)

STEP 2: ASSIGNMENT
  → For each data point x_i:
      Compute distance to ALL K centroids
      Assign x_i to the cluster with the NEAREST centroid
      cluster(x_i) = argmin_k  dist(x_i, centroid_k)

STEP 3: UPDATE
  → For each cluster k:
      Recalculate centroid as the MEAN of all assigned points
      centroid_k = mean of all x_i where cluster(x_i) = k

STEP 4: REPEAT
  → Repeat Steps 2 and 3 until:
      (a) Assignments don't change, OR
      (b) Centroids move less than a threshold, OR
      (c) Maximum iterations reached

─────────────────────────────────────────────────
```

### Visual Example (2D, K=3):

```
Iteration 1:               Iteration 2:               Converged:
  *  ×  ×   ·  ·             *  ×  ×   ·  ·             *  ×  ×   ·  ·
  * [C1] ×   · [C2]·           * [C1]×   · [C2]·           * [C1]×   · [C2]·
  *  ×  ×   ·  ·             *  ×  ×   ·  ·             *  ×  ×   ·  ·
    ○  ○                        ○  ○                        ○  ○
    ○ [C3]○                     ○ [C3]○                     ○ [C3]○

[C1],[C2],[C3] = centroids
*, ×, ○, · = data points assigned to clusters
```

---

## 4. The Mathematics Behind K-Means

### Objective Function (WCSS)

K-Means minimizes the **Within-Cluster Sum of Squares (WCSS)**, also called **Inertia**:

```
        K     n
J  =   Σ    Σ   || x_i  -  μ_k ||²
       k=1  i=1
       (for all points x_i assigned to cluster k)

Where:
  K    = number of clusters
  x_i  = data point i
  μ_k  = centroid of cluster k
  ||·||² = squared Euclidean distance
```

### Distance Metrics

The default (and most common) distance metric is **Euclidean Distance**:

```
Euclidean:  d(a, b) = √[ Σ (a_i - b_i)² ]

Manhattan:  d(a, b) = Σ |a_i - b_i|

Cosine:     d(a, b) = 1 - (a · b) / (||a|| × ||b||)
```

### Centroid Update Rule

After each assignment step, centroids are updated as:

```
        1      n
μ_k  = ─────  Σ   x_i        (mean of all points in cluster k)
       |C_k|  i=1
       (where x_i ∈ C_k)
```

---

## 5. Choosing K — The Elbow Method & More

Choosing the right K is one of the biggest challenges in K-Means.

### 5.1 Elbow Method

Plot WCSS (inertia) vs K. The "elbow" point — where the rate of decrease slows sharply — is the optimal K.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=500, centers=4, random_state=42)

# Compute WCSS for different K values
wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('WCSS (Inertia)', fontsize=12)
plt.title('Elbow Method — Optimal K', fontsize=14)
plt.xticks(K_range)
plt.grid(True, alpha=0.3)
plt.axvline(x=4, color='red', linestyle='--', label='Optimal K=4')
plt.legend()
plt.tight_layout()
plt.savefig('elbow_curve.png', dpi=150)
plt.show()
```

---

### 5.2 Silhouette Score

Measures how similar a point is to its own cluster vs neighboring clusters. Score ranges from **-1 (bad) to +1 (perfect)**.

```python
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

silhouette_scores = []
K_range = range(2, 11)  # Silhouette needs at least 2 clusters

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette Score = {score:.4f}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, 'rs-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Silhouette Analysis', fontsize=14)
plt.xticks(K_range)
plt.grid(True, alpha=0.3)
best_k = K_range[silhouette_scores.index(max(silhouette_scores))]
plt.axvline(x=best_k, color='green', linestyle='--', label=f'Best K={best_k}')
plt.legend()
plt.tight_layout()
plt.show()
```

---

### 5.3 Gap Statistic

Compares WCSS of your data against randomly generated uniform data. The K with the largest gap is optimal.

```python
# Simplified Gap Statistic
def gap_statistic(X, k_max=10, n_refs=10):
    gaps = []
    for k in range(1, k_max + 1):
        # Cluster actual data
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        actual_wcss = np.log(km.inertia_)

        # Cluster random reference datasets
        ref_wcss = []
        for _ in range(n_refs):
            random_data = np.random.uniform(X.min(0), X.max(0), X.shape)
            km_ref = KMeans(n_clusters=k, n_init=5, random_state=42)
            km_ref.fit(random_data)
            ref_wcss.append(np.log(km_ref.inertia_))

        gap = np.mean(ref_wcss) - actual_wcss
        gaps.append(gap)

    return gaps
```

---

## 6. K-Means Variants

### K-Means++ (Better Initialization)

Standard K-Means places initial centroids randomly, which can lead to poor results. **K-Means++** spreads them out intelligently:

```
1. Pick first centroid randomly from data
2. For each remaining centroid:
   - Compute D(x) = distance from x to nearest existing centroid
   - Pick next centroid with probability ∝ D(x)²
   - (Points far from existing centroids are more likely to be chosen)
3. Run standard K-Means
```

```python
# Use K-Means++ (default in sklearn)
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
```

### Mini-Batch K-Means (Faster for Large Datasets)

Uses random mini-batches instead of the full dataset for each update — much faster with minimal accuracy loss.

```python
from sklearn.cluster import MiniBatchKMeans

mbkm = MiniBatchKMeans(
    n_clusters=4,
    batch_size=256,
    n_init=10,
    random_state=42
)
mbkm.fit(X_large)
```

### K-Medoids (Robust to Outliers)

Instead of using the mean as centroid, K-Medoids uses an actual data point (the **medoid**). More robust to outliers.

```python
# pip install scikit-learn-extra
from sklearn_extra.cluster import KMedoids

kmedoids = KMedoids(n_clusters=4, random_state=42)
kmedoids.fit(X)
```

---

## 7. Implementation from Scratch

Understanding the algorithm by implementing it manually:

```python
import numpy as np
import matplotlib.pyplot as plt

class KMeansScratch:
    def __init__(self, K=3, max_iters=100, tol=1e-4, random_state=42):
        self.K = K
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_history = []

    def _init_centroids(self, X):
        """K-Means++ initialization"""
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        # Pick first centroid randomly
        idx = np.random.randint(0, n_samples)
        centroids = [X[idx]]

        for _ in range(1, self.K):
            # Compute distances to nearest centroid
            distances = np.array([
                min(np.linalg.norm(x - c) ** 2 for c in centroids)
                for x in X
            ])
            # Pick next centroid with probability proportional to D²
            probs = distances / distances.sum()
            idx = np.random.choice(n_samples, p=probs)
            centroids.append(X[idx])

        return np.array(centroids)

    def _assign_clusters(self, X):
        """Assign each point to nearest centroid"""
        distances = np.array([
            [np.linalg.norm(x - c) for c in self.centroids]
            for x in X
        ])
        return distances.argmin(axis=1)

    def _update_centroids(self, X, labels):
        """Recompute centroids as cluster means"""
        return np.array([
            X[labels == k].mean(axis=0) if (labels == k).sum() > 0
            else self.centroids[k]
            for k in range(self.K)
        ])

    def _compute_inertia(self, X, labels):
        """Compute WCSS"""
        return sum(
            np.linalg.norm(X[i] - self.centroids[labels[i]]) ** 2
            for i in range(len(X))
        )

    def fit(self, X):
        self.centroids = self._init_centroids(X)

        for iteration in range(self.max_iters):
            # Assignment step
            labels = self._assign_clusters(X)

            # Update step
            new_centroids = self._update_centroids(X, labels)

            # Track inertia
            inertia = self._compute_inertia(X, labels)
            self.inertia_history.append(inertia)

            # Check convergence
            shift = np.linalg.norm(new_centroids - self.centroids)
            if shift < self.tol:
                print(f"Converged at iteration {iteration + 1}")
                break

            self.centroids = new_centroids

        self.labels = labels
        self.inertia_ = self.inertia_history[-1]
        return self

    def predict(self, X):
        return self._assign_clusters(X)

    def fit_predict(self, X):
        return self.fit(X).labels


# ── Test It ───────────────────────────────────────────
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

km = KMeansScratch(K=4, random_state=42)
labels = km.fit_predict(X)

print(f"Final Inertia: {km.inertia_:.2f}")
print(f"Centroids:\n{km.centroids}")

# Plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(km.centroids[:, 0], km.centroids[:, 1],
            c='red', marker='X', s=200, linewidths=2, label='Centroids')
plt.title('K-Means from Scratch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(km.inertia_history, 'b-o')
plt.xlabel('Iteration')
plt.ylabel('Inertia (WCSS)')
plt.title('Convergence Curve')
plt.grid(True)

plt.tight_layout()
plt.savefig('kmeans_scratch.png', dpi=150)
plt.show()
```

---

## 8. K-Means with Scikit-Learn

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import numpy as np

# ── 1. Generate Data ──────────────────────────────────
X, y_true = make_blobs(n_samples=500, centers=4,
                        cluster_std=0.8, random_state=42)

# ── 2. Preprocess — ALWAYS scale for K-Means ──────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 3. Fit K-Means ────────────────────────────────────
kmeans = KMeans(
    n_clusters=4,
    init='k-means++',    # Smart initialization
    n_init=10,           # Run 10 times, pick best
    max_iter=300,        # Max iterations per run
    tol=1e-4,            # Convergence tolerance
    random_state=42,
    verbose=0
)

kmeans.fit(X_scaled)

# ── 4. Results ────────────────────────────────────────
labels     = kmeans.labels_         # Cluster assignments
centroids  = kmeans.cluster_centers_ # Centroid coordinates
inertia    = kmeans.inertia_         # WCSS value
iterations = kmeans.n_iter_          # Iterations to converge

print(f"Cluster Labels: {np.unique(labels)}")
print(f"Inertia: {inertia:.2f}")
print(f"Converged in {iterations} iterations")
print(f"Centroids shape: {centroids.shape}")

# ── 5. Predict new data ───────────────────────────────
new_points = np.array([[1.0, 2.0], [-1.0, -2.0]])
new_scaled = scaler.transform(new_points)
new_labels = kmeans.predict(new_scaled)
print(f"New point clusters: {new_labels}")

# ── 6. Transform to cluster distances ─────────────────
distances = kmeans.transform(X_scaled)   # (n_samples, K) distance matrix
print(f"Distance matrix shape: {distances.shape}")
```

---

## 9. Evaluating Clustering Quality

Since clustering is unsupervised (no ground truth labels), evaluation requires special metrics:

```python
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

labels = kmeans.fit_predict(X_scaled)

# ── Without ground truth ──────────────────────────────

# 1. Silhouette Score: -1 to 1 (higher = better)
sil = silhouette_score(X_scaled, labels)
print(f"Silhouette Score:        {sil:.4f}  (higher is better)")

# 2. Calinski-Harabasz Index: higher = better
ch = calinski_harabasz_score(X_scaled, labels)
print(f"Calinski-Harabasz Index: {ch:.2f}  (higher is better)")

# 3. Davies-Bouldin Index: lower = better
db = davies_bouldin_score(X_scaled, labels)
print(f"Davies-Bouldin Index:    {db:.4f}  (lower is better)")

# ── With ground truth (if available) ─────────────────

# 4. Adjusted Rand Index: -1 to 1 (1 = perfect)
ari = adjusted_rand_score(y_true, labels)
print(f"Adjusted Rand Index:     {ari:.4f}  (1 = perfect)")

# 5. Normalized Mutual Information: 0 to 1 (1 = perfect)
nmi = normalized_mutual_info_score(y_true, labels)
print(f"NMI Score:               {nmi:.4f}  (1 = perfect)")
```

---

## 10. Visualizing Clusters

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── 2D Cluster Plot ───────────────────────────────────
def plot_clusters(X, labels, centroids, title='K-Means Clusters'):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels,
                          cmap='tab10', alpha=0.6, s=30)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                c='black', marker='X', s=250,
                linewidths=1.5, zorder=5, label='Centroids')
    plt.colorbar(scatter, label='Cluster')
    plt.title(title, fontsize=14)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ── PCA for high-dimensional data ─────────────────────
def plot_pca_clusters(X, labels, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                          c=labels, cmap='tab10', alpha=0.6, s=30)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'PCA — Explained Variance: {explained:.1%}', fontsize=14)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.tight_layout()
    plt.show()

# ── t-SNE for complex high-dimensional data ───────────
def plot_tsne_clusters(X, labels):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                          c=labels, cmap='tab10', alpha=0.6, s=30)
    plt.colorbar(scatter, label='Cluster')
    plt.title('t-SNE Visualization of Clusters', fontsize=14)
    plt.tight_layout()
    plt.show()
```

---

## 11. Real-World Applications

### Customer Segmentation
```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sample customer data
data = {
    'annual_income': [15, 16, 17, 18, 19, 70, 71, 72, 73, 74,
                      120, 122, 124, 126, 128, 80, 82, 84, 86, 88],
    'spending_score': [39, 81, 6, 77, 40, 61, 21, 50, 23, 75,
                       97, 50, 85, 3, 48, 41, 7, 50, 50, 50],
    'age':           [19, 21, 20, 23, 31, 29, 35, 30, 35, 40,
                      40, 46, 45, 60, 55, 48, 52, 39, 37, 55]
}
df = pd.DataFrame(data)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Find optimal K
wcss = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled).inertia_
        for k in range(1, 11)]

# Fit with optimal K
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
df['Segment'] = kmeans.fit_predict(X_scaled)

# Analyze segments
print(df.groupby('Segment').mean().round(2))

# Name segments based on characteristics
segment_names = {
    0: 'Low Income, Low Spenders',
    1: 'Low Income, High Spenders',
    2: 'Mid Income, Mid Spenders',
    3: 'High Income, Low Spenders',
    4: 'High Income, High Spenders'
}
```

### Image Color Quantization
```python
import cv2
import numpy as np
from sklearn.cluster import KMeans

def quantize_image(image_path, n_colors=8):
    """Reduce image to N dominant colors using K-Means"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape to (n_pixels, 3)
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)

    # Cluster pixels by color
    kmeans = KMeans(n_clusters=n_colors, n_init=3, random_state=42)
    labels = kmeans.fit_predict(pixels)

    # Replace each pixel with its centroid color
    quantized_pixels = kmeans.cluster_centers_[labels]
    quantized_img = quantized_pixels.reshape(img_rgb.shape).astype(np.uint8)

    print(f"Dominant Colors:\n{kmeans.cluster_centers_.astype(int)}")
    return quantized_img, kmeans.cluster_centers_
```

---

## 12. Related Clustering Algorithms

| Algorithm | Type | Best For | Key Difference |
|---|---|---|---|
| **K-Means** | Centroid-based | Spherical, equal-size clusters | Fast, scalable |
| **K-Means++** | Centroid-based | Same + better init | Smarter initialization |
| **DBSCAN** | Density-based | Arbitrary shapes, noise | No need to specify K |
| **Hierarchical** | Agglomerative | Tree structure, dendrogram | Merges/splits clusters |
| **Gaussian Mixture** | Probabilistic | Soft assignments, ellipsoidal | Probability-based |
| **Mean Shift** | Density-based | Non-spherical clusters | Automatically finds K |
| **Spectral** | Graph-based | Complex shapes | Uses graph Laplacian |
| **BIRCH** | Tree-based | Very large datasets | Memory-efficient |

```python
# Quick comparison
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# DBSCAN — no need to specify K, handles noise
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
print(f"DBSCAN clusters found: {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)}")

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
agg_labels = agg.fit_predict(X_scaled)

# Gaussian Mixture Model
gmm = GaussianMixture(n_components=4, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
proba = gmm.predict_proba(X_scaled)  # Soft assignments
```

---

## 13. Full Project Examples

### Complete Customer Segmentation Pipeline

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ── 1. Load & Explore Data ────────────────────────────
df = pd.read_csv('customers.csv')
print(df.head())
print(df.describe())
print(df.isnull().sum())

# ── 2. Select & Scale Features ───────────────────────
features = ['annual_income', 'spending_score', 'age', 'purchase_frequency']
X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 3. Find Optimal K ─────────────────────────────────
wcss, silhouette = [], []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    wcss.append(km.inertia_)
    silhouette.append(silhouette_score(X_scaled, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(K_range, wcss, 'bo-', linewidth=2)
ax1.set_title('Elbow Method')
ax1.set_xlabel('K')
ax1.set_ylabel('WCSS')
ax1.grid(True, alpha=0.3)

ax2.plot(K_range, silhouette, 'rs-', linewidth=2)
ax2.set_title('Silhouette Analysis')
ax2.set_xlabel('K')
ax2.set_ylabel('Silhouette Score')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('optimal_k.png', dpi=150)
plt.show()

# ── 4. Fit Final Model ────────────────────────────────
OPTIMAL_K = 5
kmeans = KMeans(n_clusters=OPTIMAL_K, init='k-means++', n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# ── 5. Visualize with PCA ─────────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centroids_pca = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(10, 7))
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
for i in range(OPTIMAL_K):
    mask = df['Cluster'] == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=colors[i], label=f'Cluster {i}', alpha=0.6, s=40)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            c='black', marker='X', s=300, label='Centroids', zorder=5)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
plt.title(f'Customer Segments — K={OPTIMAL_K}', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('customer_segments.png', dpi=150)
plt.show()

# ── 6. Profile Segments ───────────────────────────────
profile = df.groupby('Cluster')[features].mean().round(2)
print("\nCluster Profiles:")
print(profile)
profile.to_csv('cluster_profiles.csv')
```

---

## 14. Advantages & Limitations

### ✅ Advantages
- **Simple & fast** — easy to understand and implement
- **Scalable** — works well with large datasets (especially Mini-Batch)
- **Guaranteed convergence** — always converges (to a local minimum)
- **Versatile** — useful in many domains (images, text, customers)
- **Interpretable** — cluster centroids are meaningful

### ❌ Limitations
- **Must specify K** — you need to know the number of clusters beforehand
- **Sensitive to initialization** — poor initial centroids can lead to suboptimal results (use K-Means++)
- **Assumes spherical clusters** — struggles with elongated or irregular shapes
- **Sensitive to outliers** — extreme values can distort centroids (use K-Medoids instead)
- **Sensitive to scale** — always normalize features before clustering
- **Hard assignments** — each point belongs to exactly one cluster (no fuzzy membership)
- **Not suitable for non-convex shapes** — use DBSCAN or Spectral Clustering instead

---

## 15. Tips & Best Practices

### ✅ Do's
- **Always scale features** — use `StandardScaler` before K-Means (distance-based!)
- **Use K-Means++** — `init='k-means++'` always (it's the default in sklearn)
- **Run multiple times** — use `n_init=10` or more for stability
- **Use both Elbow + Silhouette** — don't rely on just one method for K selection
- **Visualize your clusters** — use PCA or t-SNE to understand the structure
- **Profile cluster characteristics** — compute mean of each feature per cluster
- **Handle outliers first** — remove or cap extreme values before clustering

---

## 16. Common Mistakes to Avoid

| ❌ Mistake | ✅ Correct Approach |
|---|---|
| Not scaling features | Always apply `StandardScaler` first |
| Running only once (n_init=1) | Use `n_init=10` minimum |
| Choosing K by guessing | Use Elbow method + Silhouette score |
| Using K-Means on non-spherical clusters | Use DBSCAN or Spectral Clustering |
| Ignoring outliers | Remove/clip outliers before fitting |
| Expecting globally optimal solution | K-Means finds local optimum; use multiple restarts |
| Using on categorical data | K-Means needs numerical, continuous features |
| Not validating with domain knowledge | Always check if clusters make business sense |

---

## 17. Resources & References

### 📚 Books
- *The Elements of Statistical Learning* — Hastie, Tibshirani, Friedman
- *Pattern Recognition and Machine Learning* — Christopher Bishop
- *Introduction to Machine Learning with Python* — Andreas Müller & Sarah Guido

### 🌐 Documentation
- [Scikit-Learn: K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Scikit-Learn: Clustering User Guide](https://scikit-learn.org/stable/modules/clustering.html)
- [Scikit-Learn: Clustering Metrics](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)

### 📄 Key Papers
- MacQueen (1967) — *Some Methods for Classification and Analysis of Multivariate Observations* (original K-Means)
- Arthur & Vassilvitskii (2007) — *K-Means++: The Advantages of Careful Seeding*
- Sculley (2010) — *Web-Scale K-Means Clustering* (Mini-Batch K-Means)

---

## 👨‍💻 Author

**Syed Faizan Ali**  
Computer Science Student | Machine Learning Enthusiast  
📍 Rahim Yar Khan, Punjab, Pakistan

[![GitHub](https://img.shields.io/badge/GitHub-SyedFaizanAlii-black?logo=github)](https://github.com/SyedFaizanAlii)
[![Kaggle](https://img.shields.io/badge/Kaggle-syedfaizanalii-blue?logo=kaggle)](https://www.kaggle.com/syedfaizanalii)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://in/syed-faizan-ali-70b164306)

---

> ⭐ *If this repository helped you learn K-Means Clustering, please give it a star!*  
> 🍴 *Feel free to fork, contribute, and share with fellow learners.*

---

*Last Updated: 2026 | License: MIT*
