import decision_tree
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

X = np.load('./data/X_te_centroid.npy')
y = np.load('./data/Y_te_centroid.npy')
X, X_test, y, y_test = train_test_split( X, y, test_size=0.98, random_state=42)
X, X_test, y, y_test = train_test_split( X, y, test_size=0.3, random_state=42)
# print (X.shape)
# print (y[:10])

tree = decision_tree.Tree(max_depth=10,min_criterion_improve=1e-3,criterion=decision_tree.entropy,is_classification=True)

tree.fit(X ,y)
y_pred = tree.predict(X_test)

print('accuracy:',accuracy_score(y_test, y_pred))
