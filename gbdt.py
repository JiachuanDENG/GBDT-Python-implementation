from decision_tree import Tree
import numpy as np
import math
from tqdm import tqdm

class BionominalLoss(object):
    def __call__(self,y,F):
        """
        y: np.array[N,] labels \in {0,1}
        F: np.array[N,] logit(prob)
        return : scalar
        """
        return sum(-1*y*F + np.log(1+np.exp(F)))
    def get_residual(self,y,F):
        """
        y: np.array[N,] labels \in {0,1}
        F: np.array[N,] logit(prob)
        return :
            residual: np.array[N,]
        """
        return y - np.exp(F)/(1+np.exp(F))

    def update_terminal_region(self,tree,X,y,residual):
        """
        update tree node value based on one step newton method
        """

        # build map between each node in tree with samples in X
        leaf_nodes = []
        for idx in range(X.shape[0]):
            node = tree.get_sample_belongNode(X[idx]) # get the leaf node that x belongs to
            node.update_region_dataidx(idx)
            leaf_nodes.append(node)

        # update node value based on one step Newton method
        for node in leaf_nodes:
            subset_idxes = np.array(list(node.data_idxes)).astype(np.int16)
            y_subset = y[subset_idxes]
            residual_subset = residual[subset_idxes]

            nominator = sum(residual_subset)
            denominator = sum((y_subset-residual_subset)*(1-y_subset+residual_subset))
            # update value
            node.value = nominator/(denominator)





class GBDT(object):
    """
    gbdt for classification task
    """
    def __init__(self,n_estimators,loss=BionominalLoss,max_depth=3, min_criterion_improve=0,min_samples_leaf=1,learning_rate=0.5):
        self.n_estimators = n_estimators
        self.loss = loss()
        self.max_depth = max_depth
        self.min_criterion_improve = min_criterion_improve
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.trees = []

    def get_F(self,X,y=None):
        """
        get logit(prob) value given x
        X: np.array[N,M]
        return:
            F: [N,]float value
        """
        if len(self.trees) == 0:
            # print ('init')
            assert not (y is None), 'tree0 need y to initialize,y:{}'.format(y)
            pos = y.sum()
            neg = y.shape[0] - pos
            F = np.zeros([X.shape[0],])
            F[:] = math.log(pos / neg)
            # print (pos,neg,F)
            return F
        else:
            F = np.zeros([X.shape[0],])
            for tree in self.trees:
                F += tree.predict(X)
            # print (F)
            return F


    def fit (self,X,y):
        """
        X: np.array [N,M]
        y: np.array [N,]
        """

        for estimator_idx in tqdm(range(self.n_estimators)):
            print ('esitmator',estimator_idx)
            if estimator_idx == 0:
                F = self.get_F(X,y)
            else:
                F = self.get_F(X)
            # print (F)
            residual = self.loss.get_residual(y,F)
            # assert y.shape[0] == residual.shape[0],'y shape {}, residual shape {}'.format(y.shape,residual.shape)
            tree = Tree(max_depth=self.max_depth,min_criterion_improve=self.min_criterion_improve,\
            min_samples_leaf = self.min_samples_leaf)
            tree.fit(X,residual)
            self.loss.update_terminal_region(tree,X,y,residual)
            self.trees.append(tree)

    def predict(self,X):
        """
        X : np.array[N,M]
        """
        F = self.get_F(X)
        # print (F)
        pred = np.zeros([F.shape[0],])
        pred[F>0] = 1
        return pred
