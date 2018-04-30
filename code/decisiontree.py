import pandas as pd
import graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support as score

loaded_data = pd.read_csv('../data/arc.csv', sep= ',', header=0)

X = loaded_data.values[:, 3:-1].tolist() # Select from column 4 till last but 1
Y = loaded_data.values[:,-1].tolist() # Selecr last column

# Update Y to contain either 1s or 0s
for index, item in enumerate(Y):
  if item >= 1:
    Y[index] = 1
  else:
    Y[index] = 0
  # End of if
# End of for

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
precision, recall, fscore, support = score(Y, Y_pred, average='micro')
print('Precision: ', precision)
print('Recall: ', recall)
print('F1-Score: ', fscore)
print('Support: ', support)

# Plot graph
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data) 
graph.render("Output")