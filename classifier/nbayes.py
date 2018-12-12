from math import sqrt

import numpy as np
from sklearn.metrics import confusion_matrix


class NBayes():
    def __init__(self, lamda=1):
        self.lamda = lamda  # for smoothing
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

        self.tot_train_num = len(y_train)
        self.y_class_indexes = dict(zip(self.y_kinds, [[]] * len(self.y_kinds)))
        for i in range(self.tot_train_num):
            self.y_class_indexes[y_train[i]].append(i)

        self._calcu_prior_probability()
        self._calcu_conditional_probability(X_train)

        self.built = True

    def _calcu_prior_probability(self):
        # calculate prior probability
        self.prior_y = dict(zip(self.y_kinds, [0] * len(self.y_kinds)))
        for y_v in self.prior_y:
            # smoothing
            self.prior_y[y_v] = float(len(self.y_class_indexes[y_v]) + self.lamda) / (
                        self.tot_train_num + self.lamda * len(self.y_kinds))

    def _calcu_conditional_probability(self, X_train):
        # calculate conditional probability
        self.cond_prob = dict()
        for y_v in self.y_kinds:
            self.cond_prob[y_v] = dict(zip(self.attr_list, [[]] * len(self.attr_list)))
            for attr in self.attr_list:
                cur_y_attr = X_train[self.y_class_indexes[y_v], self.attr_position_map[attr]]
                if (not self.attr_is_discrete_map[attr]):
                    self.cond_prob[y_v][attr] = [np.mean(cur_y_attr), np.std(cur_y_attr)]
                else:
                    self.cond_prob[y_v][attr] = dict(
                        zip(self.attr_discrete_values[attr], [0] * len(self.attr_discrete_values[attr])))
                    for attr_v in cur_y_attr:
                        self.cond_prob[y_v][attr][attr_v] += 1
                    for attr_v in self.attr_discrete_values[attr]:
                        self.cond_prob[y_v][attr][attr_v] = float(self.cond_prob[y_v][attr][attr_v] + self.lamda) / (
                                    len(cur_y_attr) + self.lamda * len(self.attr_discrete_values[attr]))

    def _calcu_gauss_prob(self, x_v, mean, stdev):
        exponent = np.exp(-(np.power(x_v - mean, 2)) / (2 * np.power(stdev, 2)))
        gauss_prob = (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent
        return gauss_prob

    def predict_one_record(self, x):
        y_preds = dict(zip(self.y_kinds, [1] * len(self.y_kinds)))
        for y_v in self.y_kinds:
            for attr in self.attr_list:
                x_attr_v = x[self.attr_position_map[attr]]
                if self.attr_is_discrete_map[attr]:
                    y_preds[y_v] *= self.cond_prob[y_v][attr][x_attr_v]
                else:
                    mean, std = self.cond_prob[y_v][attr]
                    y_preds[y_v] *= self._calcu_gauss_prob(x_attr_v, mean, std)
        preds_sum = sum(y_preds.values())

        y_pred_class = -1
        y_max_prob = -1
        for y_v in y_preds:
            y_preds[y_v] = y_preds[y_v] / preds_sum
            if y_preds[y_v] > y_max_prob:
                y_max_prob = y_preds[y_v]
                y_pred_class = y_v
        return y_pred_class, y_preds

    def predict(self, X_test):
        if not self.built:
            print("You should build the NBayes first by calling the 'fit' method with some train samples.")
            return None

        y_predicts = []
        for x in X_test:
            y_predicts.append(self.predict_one_record(x)[0])
        return y_predicts

    def evaluate(self, X_test, y_test, detailed_result=0):
        y_predict = self.predict(X_test)
        return self._calculate_metrics(y_predict, y_test, detailed_result)

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

    bank = pd.read_csv('./data/bank.csv')
    X = np.array(bank.ix[:,bank.columns[0:-1]], dtype=object)
    y = np.array(bank.ix[:,bank.columns[-1]])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    attr_list = ['age', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    categorical_attris = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

    nbayes = NBayes()
    nbayes.fit(X_train, y_train, attr_list, attr_is_discrete=[x in categorical_attris for x in attr_list])


    nbayes.evaluate(X_train, y_train)