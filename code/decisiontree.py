import pandas as pd
import graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
loaded_data = pd.read_csv('../data/arc.csv', sep= ',', header=0)

# print('Dataset Lenght:: ', len(loaded_data))
# print('Dataset Shape:: ', loaded_data.shape)
# print('Dataset:: \n', loaded_data.head())

X = loaded_data.values[:, 3:-1].tolist() # Select from column 4 till last but 1
Y = loaded_data.values[:,-1].tolist() # Selecr last column

# sklearn accepts only numerical features for decision tree
# Update X with only numerical values
# for index, innerlist in enumerate(X):
#   for idx, listitem in enumerate(innerlist):
#     # Replace the first column with value 1 as they always contains same value
#     # Replace the third column with index  value  as they always contains unique value
#     innerlist[0] = 1
#     innerlist[2] = index
#     X[index] = innerlist
  # End of for
# End of for

# Update Y to contain either 1s or 0s
for index, item in enumerate(Y):
  if item >= 1:
    Y[index] = 1
  else:
    Y[index] = 0
  # End of if
# End of for

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state = 100)

# # Use GINI Index to measure the quality of attributes split
# # Pick the attribute with BEST GINI Index for splitting
# clf = DecisionTreeClassifier(criterion='gini', splitter='best', random_state=100)
# clf = clf.fit(X_train, Y_train)

# Y_pred = clf.predict(X_test)

clf = DecisionTreeClassifier(criterion='gini', splitter='best', random_state=100)
clf = clf.fit(X, Y)

Y_pred = clf.predict(X)

print('Actual bug class values:')
for value in Y:
  print(value, end=" ")

print('\nPredicted bug class values:')
for value in Y_pred:
  print(value, end=" ")

print('\nAccuracy: ', accuracy_score(Y, Y_pred) * 100)

# Plot graph
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data) 
graph.render("Output")