import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import skfuzzy as fuzz
# from sklearn.model_selection import train_test_split

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

DATASET_PATH='../data/'
DATASET_NAME='arc.csv'  # Change the file name here
loaded_data = pd.read_csv(DATASET_PATH + DATASET_NAME, sep= ',', header=0)

data = loaded_data.values[:, 3:-1] # Select from column 4 till last but 1
labels = loaded_data.values[:,-1].tolist() # Selecr last column Convert numpy array to list

# Data preprocessing
# Update labels to contain either 1s or 0s
for index, item in enumerate(labels):
  if item >= 1:
    labels[index] = 1
  else:
    labels[index] = 0
  # End of if
# End of for
labels = np.asarray(labels) # Convert back to numpy array

# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=0)

n_samples, n_features = data.shape
n_digits = len(np.unique(labels))
print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


# Set up the loop
ncenters = 2
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data, ncenters, 2, error=0.005, maxiter=1000, init=None)


# Predict
# cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
#     data, 3, 2, error=0.005, maxiter=1000)

print(fuzz.cluster.cmeans_predict(data, 3, 2, error=0.005, maxiter=1000))