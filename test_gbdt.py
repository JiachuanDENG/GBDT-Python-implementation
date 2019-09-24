from gbdt import GBDT

from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split

X = np.load('./data/X_te_centroid.npy')
y = np.load('./data/Y_te_centroid.npy')
X, X_test, y, y_test = train_test_split( X, y, test_size=0.98, random_state=42)
X, X_test, y, y_test = train_test_split( X, y, test_size=0.3, random_state=42)
print (X.shape,X_test.shape)


clf = GBDT(n_estimators=10)

clf.fit(X,y)
y_pred = clf.predict(X_test)

print('accuracy:',accuracy_score(y_test, y_pred))
