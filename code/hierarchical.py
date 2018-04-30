from time import time

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from sklearn.cluster import AgglomerativeClustering
from sklearn import manifold


DATASET_PATH='../data/'
DATASET_NAME='forrest-0.8.csv'  # Change the file name here
loaded_data = pd.read_csv(DATASET_PATH + DATASET_NAME, sep= ',', header=0)

X = loaded_data.values[:, 3:-1] # Select from column 4 till last but 1
Y = loaded_data.values[:,-1].tolist() # Selecr last column

# Update Y to contain either 1s or 0s
for index, item in enumerate(Y):
  if item >= 1:
    Y[index] = 1
  else:
    Y[index] = 0
  # End of if
# End of for
Y = np.asarray(Y) # Convert back to numpy array

# digits = datasets.load_digits(n_class=10)
# X = digits.data
# Y = digits.target
n_samples, n_features = X.shape
n_digits = len(np.unique(Y))

np.random.seed(0)

#----------------------------------------------------------------------
# Visualize the clustering
def plot_clustering(X_red, X, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(Y[i]),
                 color=cm.nipy_spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout()

#----------------------------------------------------------------------
# Dendrogram plotting
def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

#----------------------------------------------------------------------
# 2D embedding of the digits dataset
print("Computing embedding")
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
print("Done.")

# for linkage in ('ward', 'average', 'complete'):
#     clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
#     t0 = time()
#     clustering.fit(X_red)
#     print("%s : %.2fs" % (linkage, time() - t0))

#     plot_clustering(X_red, X, clustering.labels_, "%s linkage" % linkage)

clustering = AgglomerativeClustering(linkage='complete', n_clusters=n_digits)
t0 = time()
model = clustering.fit(X_red)
print("%s : %.2fs" % ('complete', time() - t0))
plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(model, labels=model.labels_)
plt.show()

# plot_clustering(X_red, X, clustering.labels_, "%s linkage" % 'complete')
# plt.show()