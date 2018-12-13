from classifier.dtree import DTC45

from math import log2
from math import sqrt
from random import randint
from queue import PriorityQueue

import numpy as np
from sklearn.metrics import confusion_matrix


class RandomForest():
    def __init__(self, tree_number=30, max_depth=100, min_samples_split=2, max_continuous_attr_splits=200):
        self.tree_number = tree_number
        self.max_tree_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_continuous_attr_splits = max_continuous_attr_splits
        self.built = False

    def fit(self, X_train, y_train, attr_list, attr_is_discrete, attr_discrete_values=None, verbose=0):

        self.y_kinds = set(y_train)
        self.attr_list = attr_list
        self.attr_position_map = dict(zip(attr_list, range(len(attr_list))))
        # if the elements in *attr_is_discrete* are [True, False] format, convert them to [1, 0] format
        if attr_is_discrete[0] in [True, False]:
            attr_is_discrete = list([1 if x == True else 0 for x in attr_is_discrete])
        self.attr_is_discrete_map = dict(zip(attr_list, attr_is_discrete))

        if attr_discrete_values != None:
            self.attr_discrete_values = attr_discrete_values
        else:
            self.attr_discrete_values = {}
            for i in range(len(attr_list)):
                attr = attr_list[i]
                if self.attr_is_discrete_map[attr]:
                    self.attr_discrete_values[attr] = set([x[i] for x in X_train])

        # bagging sampling and building all subtrees
        self.trees = []
        for i in range(self.tree_number):
            print('# Building tree %d...' % (i + 1))
            self.trees.append(self._build_tree(X_train, y_train))
        self.built = True

    def _build_tree(self, X_train, y_train):
        tot_train_num = len(y_train)
        train_indexes = [randint(0, tot_train_num - 1) for i in range(tot_train_num)]
        dtree = DTC45(max_depth=self.max_tree_depth, min_samples_split=self.min_samples_split,
                      max_continuous_attr_splits=self.max_continuous_attr_splits)
        dtree.fit(X_train=X_train[train_indexes, :], y_train=y_train[train_indexes], attr_list=self.attr_list,
                  attr_is_discrete=[self.attr_is_discrete_map[attr] for attr in self.attr_list], attr_discrete_values=self.attr_discrete_values, verbose=0)
        return dtree

    def predict(self, X_test):
        if not self.built:
            print("You should build the RandomForest first by calling the 'fit' method with some train samples.")
            return None

        y_predicts_tot = []
        for tree in self.trees:
            y_pred = tree.predict(X_test)
            y_predicts_tot.append(y_pred)
        y_predicts_tot = np.array(y_predicts_tot)

        y_preds = []
        for i in range(len(X_test)):
            y_value_dict = dict(zip(self.y_kinds, [0]*len(self.y_kinds)))
            for y_v in y_predicts_tot[:,i]:
                y_value_dict[y_v] += 1
            y_preds.append(max(y_value_dict, key=y_value_dict.get))

        return y_preds

    # return predict probabilities for positive label in binary classification
    def predict_proba(self, X_test):
        if not self.built:
            print("You should build the RandomForest first by calling the 'fit' method with some train samples.")
            return None

        y_predicts_tot = []
        for tree in self.trees:
            y_pred = tree.predict(X_test)
            y_predicts_tot.append(y_pred)
        y_predicts_tot = np.array(y_predicts_tot)

        y_pred_probas = []
        tot_test_num = len(X_test)
        for i in range(tot_test_num):
            y_pred_probas.append(sum(y_predicts_tot[:,i]) / tot_test_num)
        return y_pred_probas

    def evaluate(self, X_test, y_test, detailed_result=0):
        y_predict = self.predict(X_test)
        return self._calculate_metrics(y_predict, y_test, detailed_result)

    def add_new_tree(self, tree_num):
        for i in range(tree_num):
            self.trees.append(self._build_tree(X_train, y_train))

    def _calculate_metrics(self, y_pred, y_true, detailed_result):
        """ If parameter detailed_result is False or 0, only prediction accuracy (Acc) will be returned.
            Otherwise, the returned result will be confusion matrix and prediction metrics list,
             in which only [Acc] for multiple classification and [Acc, Sn, Sp, Precision, MCC] for binary classification.
        """

        y_right = [1 for (y_p, y_t) in zip(y_pred, y_true) if y_p == y_t]
        acc = len(y_right) / len(y_pred)
        if not detailed_result:
            return acc

        con_matrix = confusion_matrix(y_pred, y_true)
        if len(self.y_kinds) > 2:
            return con_matrix, [acc]
        else:
            tn = con_matrix[0][0]
            fp = con_matrix[0][1]
            fn = con_matrix[1][0]
            tp = con_matrix[1][1]
            p = tp + fn
            n = tn + fp
            sn = tp / p if p > 0 else None
            sp = tn / n if n > 0 else None
            pre = (tp) / (tp + fp) if (tp + fp) > 0 else None
            mcc = 0
            tmp = sqrt(tp + fp) * sqrt(tp + fn) * sqrt(tn + fp) * sqrt(tn + fn)
            if tmp != 0:
                mcc = (tp * tn - fp * fn) / tmp
            return con_matrix, [acc, sn, sp, pre, mcc]


if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split

    bank = pd.read_csv('../data/bank.csv')
    X = np.array(bank.ix[:,bank.columns[0:-1]], dtype=object)
    y = np.array(bank.ix[:,bank.columns[-1]])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    attr_list = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    categorical_attris = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

    rf = RandomForest(tree_number=10)
    rf.fit(X_train, y_train, attr_list, attr_is_discrete=[x in categorical_attris for x in attr_list])

    print(rf.evaluate(X_train, y_train))
    print(rf.evaluate(X_test, y_test))