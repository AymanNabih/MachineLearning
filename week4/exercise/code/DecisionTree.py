from __future__ import division
import pandas as pd
import numpy as np
import math
from collections import Counter
from TreeNode import TreeNode



class DecisionTree(object):
    '''
    A decision tree class.
    '''

    def __init__(self, impurity_criterion='entropy'):
        '''
        Initialize an empty DecisionTree.
        '''

        self.root = None  # root Node
        self.feature_names = None  # string names of features (for interpreting
                                   # the tree)
        self.categorical = None  # Boolean array of whether variable is
                                 # categorical (or continuous)
        self.impurity_criterion = self._entropy \
                                  if impurity_criterion == 'entropy' \
                                  else self._gini

    def fit(self, X, y, feature_names=None):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
            - feature_names: numpy array of strings
        OUTPUT: None

        Build the decision tree.
        X is a 2 dimensional array with each column being a feature and each
        row a data point.
        y is a 1 dimensional array with each value being the corresponding
        label.
        feature_names is an optional list containing the names of each of the
        features.
        '''

        if feature_names is None or len(feature_names) != X.shape[1]:
            self.feature_names = np.arange(X.shape[1])
        else:
            self.feature_names = feature_names

        # Create True/False array of whether the variable is categorical
        is_categorical = lambda x: isinstance(x, str) or \
                                   isinstance(x, bool) or \
                                   isinstance(x, unicode)
        self.categorical = np.vectorize(is_categorical)(X[0])

        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
        OUTPUT:
            - TreeNode

        Recursively build the decision tree. Return the root node.
        '''

        node = TreeNode()
        index, value, splits = self._choose_split_index(X, y)

        if index is None or len(np.unique(y)) == 1:
            node.leaf = True
            node.classes = Counter(y)
            node.name = node.classes.most_common(1)[0][0]
        else:
            X1, y1, X2, y2 = splits
            node.column = index
            node.name = self.feature_names[index]
            node.value = value
            node.categorical = self.categorical[index]
            node.left = self._build_tree(X1, y1)
            node.right = self._build_tree(X2, y2)
        return node

    def _entropy(self, y):
        '''
        INPUT:
            - y: 1d numpy array
        OUTPUT:
            - float

        Return the entropy of the array y.
        '''
        ### YOUR CODE HERE
        pos = np.count_nonzero(y)
        print y
        ylen = len(y)
        if ylen> 0:
            pentropy=(-pos/ylen)*np.log(pos/ylen)
            nentropy=((ylen-pos)/-ylen)*np.log((ylen-pos)/ylen)
#             print pentropy
#             print nentropy
#             print pentropy+nentropy
            return pentropy+nentropy
        else:
            return 0

    def _gini(self, y):
        '''
        INPUT:
            - y: 1d numpy array
        OUTPUT:
            - float

        Return the gini impurity of the array y.
        '''

        ### YOUR CODE HERE
        pos = y.count_nonzero()
        ylen = len(y)
        if ylen> 0:
            gini = 1-((pos/ylen)**2+((ylen-pos)/ylen)**2)
            print gini
            return gini
        else:
            return 0
    def _make_split(self, X, y, split_index, split_value):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
            - split_index: int (index of feature)
            - split_value: int/float/bool/str (value of feature)
        OUTPUT:
            - X1: 2d numpy array (feature matrix for subset 1)
            - y1: 1d numpy array (labels for subset 1)
            - X2: 2d numpy array (feature matrix for subset 2)
            - y2: 1d numpy array (labels for subset 2)

        Return the two subsets of the dataset achieved by the given feature and
        value to split on.

        Call the method like this:
        >>> X1, y1, X2, y2 = self._make_split(X, y, split_index, split_value)

        X1, y1 is a subset of the data.
        X2, y2 is the other subset of the data.
        '''

        ### YOUR CODE HERE
        print X
        print y
        Z = np.hstack((X,y.reshape(-1,1)))
        print Z
        if isinstance(split_value, int) or isinstance(split_value, float):
            X1 = X[X[:, split_index] >= split_value, :] 
            X2 = X[X[:, split_index] < split_value, :]
            y1 = Z[Z[:, split_index] >= split_value, -1]
            y2 = Z[Z[:, split_index] < split_value, -1]
        else:
            X1 = X[X[:, split_index] == split_value, :]
            X2 = X[X[:, split_index] != split_value, :]
            y1 = Z[Z[:, split_index] == split_value, -1]
            y2 = Z[Z[:, split_index] != split_value, -1]
        print X1
        print X2
        print y1
        print y2
        return X1,X2,y1,y2

    def _information_gain(self, y, y1, y2):
        '''
        INPUT:
            - y: 1d numpy array
            - y1: 1d numpy array (labels for subset 1)
            - y2: 1d numpy array (labels for subset 2)
        OUTPUT:
            - float

        Return the information gain of making the given split.

        Use self.impurity_criterion(y) rather than calling _entropy or _gini
        directly.
        '''
        if len(y1) == 0:
            return self.impurity_criterion(y)-self.impurity_criterion(y2)
        if len(y2) == 0:
            return self.impurity_criterion(y)-self.impurity_criterion(12)
        ### YOUR CODE HERE
        print self.impurity_criterion(y)-(self.impurity_criterion(y1)*(len(y1)/len(y))
                                           +self.impurity_criterion(y2)*(len(y2)/len(y)))

        return self.impurity_criterion(y)-(self.impurity_criterion(y1)*(len(y1)/len(y))
                                           +self.impurity_criterion(y2)*(len(y2)/len(y)))

    def _choose_split_index(self, X, y):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
        OUTPUT:
            - index: int (index of feature)
            - value: int/float/bool/str (value of feature)
            - splits: (2d array, 1d array, 2d array, 1d array)

        Determine which feature and value to split on. Return the index and
        value of the optimal split along with the split of the dataset.

        Return None, None, None if there is no split which improves information
        gain.

        Call the method like this:
        >>> index, value, splits = self._choose_split_index(X, y)
        >>> X1, y1, X2, y2 = splits
        '''
        best_score = 0
        best_index = 0
        best_value = ''
        cols = len(X[0,:])
        for i in xrange(cols):
            for j in set(X[:,i]):
                X1, X2, y1, y2 = self._make_split(X, y, i, j)
                score = self._information_gain(y, y1, y2)
                if score > best_score:
                    best_score = score
                    best_index = i
                    best_value = j
                    bestX1, bestX2, besty1, besty2 = X1, X2, y1, y2
        if best_score > 0:
            return best_index, best_value, (bestX1, bestX2, besty1, besty2)
        else:
            return None, None, None

    def predict(self, X):
        '''
        INPUT:
            - X: 2d numpy array
        OUTPUT:
            - y: 1d numpy array

        Return an array of predictions for the feature matrix X.
        '''

        return np.apply_along_axis(self.root.predict_one, axis=1, arr=X)

    def __str__(self):
        '''
        Return string representation of the Decision Tree.
        '''
        return str(self.root)
