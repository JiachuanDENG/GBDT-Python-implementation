# GBDT-Python-implementation
GBDT (binary) classifier implemented in pure python and numpy. Could achieve equvalent performance to sklearn (in accuracy)

## Files description
* Data
  * X_te_centroid.npy: numpy array [N,100], data which can be used to test the algorithm
  * Y_te_centroid.npy: numpy array[N,], label which can be used to test the algorithm
* GBDT explaination.ipynb: detailed mathematical derivation for GBDT. It could help you a lot if you want to understand how GBDT works
* decision_tree.py: Decision tree model implemented in pure python, it will work as weak classifier of GBDT
* gbdt.py: GBDT model implemented in pure python and numpy
* test_gbdt.py: script for testing implemented algorithm (could be very slow if large dataset used, because GBDT will iterate all possible values for each feature. If dataset is large, the number of possible values for each feature could be extremely large.)

## Run the code:
python3 test_gbdt.py
