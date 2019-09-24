import numpy as np
import math

def mse_loss(y_true,y_pred):
    """
    y_true,y_pred: [N,],[N,]
    """
    assert y_true.shape[0] == y_pred.shape[0],'shape not match'
    return np.sqrt(np.sum((y_true - y_pred)**2))/y_true.shape[0]


def variance(y):
    """
    y: np.array [N,]
    """
    assert y.shape[0]>=1, 'y shape:{}'.format(y.shape)
    return np.sum((y - y.mean())*(y - y.mean()))

def entropy(y):
    """ Calculate the entropy of label array y """
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy


class Node(object):
    def __init__(self,is_leaf,value=None,feature_idx=None,feature_threshold=None):
        self.is_leaf = is_leaf
        self.value = value
        assert (self.is_leaf and (self.value!=None)) or (not self.is_leaf),'leaf node should have value'
        self.feature_idx = feature_idx
        self.feature_threshold = feature_threshold
        self.left = None # feature[i]<threshold
        self.right = None # feature[i]>=threshold
        self.data_idxes = set()
    def update_region_dataidx(self,idx):
        self.data_idxes.add(idx)

class Tree(object):
    """
    decision tree
    """
    def __init__(self,max_depth,min_criterion_improve,min_samples_leaf=1,criterion=variance,is_classification=False):

        self.max_depth = max_depth
        self.min_criterion_improve = min_criterion_improve
        self.criterion = criterion
        self.is_classification = is_classification
        self.min_samples_leaf = min_samples_leaf


    def calcluate_leaf_value(self,y):
        """
        y: np.array [N,]
        """
        if self.is_classification:
            # majority vote
            y_values,y_counts = np.unique(y, return_counts=True)
            max_count = 0
            max_idx = -1
            for i in range(y_values.shape[0]):
                if y_counts[i]>max_count:
                    max_count = y_counts[i]
                    max_idx = i

            return y_values[max_idx]
        else:
            # return average
            return y.mean()

    def criterion_improve(self,y,y1,y2):
        """
        calculate criterion improvement after split y data to y1,y2
        y,y1,y2: np.array[N,],[N1,],[N2,]
        """
        N = y.shape[0]
        N1 = y1.shape[0]
        N2 = y2.shape[0]
        assert N == (N1+N2),'shape not match'

        return self.criterion(y) - (N1/N * self.criterion(y1) + N2/N * self.criterion(y2))

    def split_data(self,X,feature_idx,threshold):
        """
        split data into 2 parts based on X[,feature_idx] value compared to threshold

        X: np.array[N,M]

        return:
            idx_1,idx_2 : np.array[N,]
            where dataset 1 is less than threshold and dataset 2 is greater than threshold
        """
        idx_1, idx_2 = np.zeros([0,]).astype(np.int16),np.zeros([0,]).astype(np.int16)
        for i in range(X.shape[0]):
            if X[i,feature_idx] < threshold:
                idx_1 = np.hstack((idx_1,np.array([i])))
            else:
                idx_2 = np.hstack((idx_2,np.array([i])))

        return idx_1, idx_2



    def build_tree (self, depth,X,y):
        """
        X: np.array[N,M]
        y: np.array[N,]
        """
        assert X.ndim == 2, 'X:{}'.format(X.shape)

        if (depth < self.max_depth) and (y.shape[0] > self.min_samples_leaf):

            largest_improvement = 0
            feature_nums = X.shape[1]
            select_feature_idx,select_feature_value = -1,-1
            select_idx_1,select_idx_2 = None,None

            for feature_idx in range(feature_nums):
                # print (X[:,feature_idx])
                feature_values = set(X[:,feature_idx])
                for feature_value in feature_values:
                    idx_1,idx_2 = self.split_data(X,feature_idx,feature_value)
                    if idx_1.shape[0]>0 and idx_2.shape[0]>0:
                        criterion_improvement = self.criterion_improve(y,y[idx_1],y[idx_2])
                        if criterion_improvement > largest_improvement:
                            largest_improvement = criterion_improvement
                            select_feature_idx = feature_idx
                            select_feature_value = feature_value
                            select_idx_1 = idx_1
                            select_idx_2 = idx_2

            if largest_improvement > self.min_criterion_improve:

                X1,X2 = X[select_idx_1], X[select_idx_2]
                y1,y2 = y[select_idx_1], y[select_idx_2]
                # print(X1.shape,select_idx_1)
                # assert X1.ndim ==2, 'X:{},X1:{},select_idx_1: {}'.format(X.shape,X1.shape,select_idx_1)
                # assert X2.ndim ==2, 'X2:{},select_idx_2: {}'.format(X2.shape,select_idx_2)
                tree_node = Node(is_leaf=False,feature_idx=select_feature_idx,feature_threshold=select_feature_value)
                tree_node.left = self.build_tree(depth+1,X1,y1)
                tree_node.right = self.build_tree(depth+1,X2,y2)
                return tree_node

            else:
                # treat as tree leaf
                return Node(is_leaf = True, value = self.calcluate_leaf_value(y))


        else:
            return Node(is_leaf = True, value = self.calcluate_leaf_value(y))

    def pred_sample(self,x,node=None):
        """
        x: np.array[M,]
        """
        if node == None:
            node = self.root

        if node.is_leaf:
            return node.value

        if x[node.feature_idx] < node.feature_threshold:
            return self.pred_sample(x,node.left)
        else:
            return self.pred_sample(x,node.right)

    def get_sample_belongNode(self,x,node=None):
        """
        x: np.array[M,]
        """
        if node == None:
            node = self.root

        if node.is_leaf:
            return node

        if x[node.feature_idx] < node.feature_threshold:
            return self.get_sample_belongNode(x,node.left)
        else:
            return self.get_sample_belongNode(x,node.right)


    def predict(self,X):
        """
        X: np.array[N,M]
        return:
            pred: np.array[N,]
        """
        pred = np.zeros([X.shape[0],])
        for i in range(X.shape[0]):
            pred[i] = self.pred_sample(X[i,:])
        return pred


    def fit(self,X,y):
        print ('fitting decision tree...')
        self.root = self.build_tree(0,X,y)
