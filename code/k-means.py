from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from matplotlib.pyplot import cm

np.random.seed(42)
DATASET_PATH='../data/'
DATASET_NAME='berek.csv'  # Change the file name here
loaded_data = pd.read_csv(DATASET_PATH + DATASET_NAME, sep= ',', header=0)

data = loaded_data.values[:, 3:-1] # Select from column 4 till last but 1
labels = loaded_data.values[:,-1].tolist() # Selecr last column Convert numpy array to list

# Data preprocessing
data = scale(data) # Standardize a dataset along any axis

# Update labels to contain either 1s or 0s
for index, item in enumerate(labels):
  if item >= 1:
    labels[index] = 1
  else:
    labels[index] = 0
  # End of if
# End of for
labels = np.asarray(labels) # Convert back to numpy array

n_samples, n_features = data.shape
n_digits = len(np.unique(labels))
print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))

sample_size = n_samples

# digits = load_digits()
# print("data before:", digits.data)
# data = scale(digits.data)
# print("data after:", data)
# print(len(digits.data), ': ', digits.data)
# print(len(digits.target), ': ', digits.target)
# n_samples, n_features = data.shape
# n_digits = len(np.unique(digits.target))
# labels = digits.target

# sample_size = 300

# print("n_digits: %d, \t n_samples %d, \t n_features %d"
#       % (n_digits, n_samples, n_features))


print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))


bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=data)

print(82 * '_')

# #############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=n_digits).fit_transform(data)
kmeans = KMeans(init='random', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the ' + DATASET_NAME + ' dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
