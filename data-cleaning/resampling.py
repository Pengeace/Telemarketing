from math import sqrt
from sklearn.metrics import roc_curve, auc
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from classifier.dtree import DTC45
from classifier.nbayes import NBayes

def calc_metrics(y_label, y_proba):
    con_matrix = confusion_matrix(y_label, [1 if x >= 0.5 else 0 for x in y_proba])
    TN = float(con_matrix[0][0])
    FP = float(con_matrix[0][1])
    FN = float(con_matrix[1][0])
    TP = float(con_matrix[1][1])
    P = TP + FN
    N = TN + FP
    Sn = TP / P
    Sp = TN / N
    Acc = (TP + TN) / (P + N)
    Avc = (Sn + Sp) / 2
    # Pre = (TP) / (TP + FP)
    MCC = 0
    tmp = sqrt((TP + FP) * (TP + FN)) * sqrt((TN + FP) * (TN + FN))
    if tmp != 0:
        MCC = (TP * TN - FP * FN) / tmp
    fpr, tpr, thresholds = roc_curve(y_label, y_proba)
    AUC = auc(fpr, tpr)
    return Sn, Sp, Acc, Avc, MCC, AUC

cur_attrs = ['age', 'job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'campaign',
             'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m',
             'nr.employed']
categorical_attris =  ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

bank = pd.read_csv('../data/bank.csv')
X = np.array(bank.ix[:, cur_attrs], dtype=object)
y = np.array(bank.ix[:, bank.columns[-1]])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


rf = RandomForestClassifier(50)
rf.fit(X_train, y_train)
print('RF',calc_metrics(y_test,rf.predict(X_test)))

nb = GaussianNB()
nb.fit(X_train, y_train)
print('NBayes',calc_metrics(y_test,nb.predict(X_test)))

lr = LogisticRegression(max_iter=500)
lr.fit(X_train, y_train)
print('LR',calc_metrics(y_test,lr.predict(X_test)))



mynb = NBayes()
mynb.fit(X_train, y_train, cur_attrs, attr_is_discrete=[x in categorical_attris for x in cur_attrs])
print('myNBayes',calc_metrics(y_test,mynb.predict(X_test)))

# dtree = DTC45()
# dtree.fit(X_train[0:-5000,:], y_train[0:-5000], cur_attrs, attr_is_discrete=[x in categorical_attris for x in cur_attrs])
# print('myDtree',calc_metrics(y_test,dtree.predict(X_test)))
# dtree.pruning(X_train[-5000:,:], y_train[-5000:])
# print('After pruning',calc_metrics(y_test,dtree.predict(X_test)))

from collections import Counter
print('# After resampling')
smote = SMOTE(ratio=0.5, sampling_strategy='minority', random_state=0)
X_res, y_res = smote.fit_sample(X_train, y_train)
for i, attr in enumerate(cur_attrs):
    if attr in categorical_attris:
        X_res[:, i] = X_res[:,i].astype('int32')
Counter(y_res)

# rf = RandomForestClassifier(50)
# rf.fit(X_res, y_res)
# print('RF',calc_metrics(y_test,rf.predict(X_test)))
#
# nb = GaussianNB()
# nb.fit(X_res, y_res)
# print('NBayes',calc_metrics(y_test,nb.predict(X_test)))
#
# lr = LogisticRegression(max_iter=500)
# lr.fit(X_res, y_res)
# print('LR',calc_metrics(y_test,lr.predict(X_test)))
#
# mynb = NBayes()
# mynb.fit(X_res, y_res, cur_attrs, attr_is_discrete=[x in categorical_attris for x in cur_attrs])
# print('myNBayes',calc_metrics(y_test,mynb.predict(X_test)))
#
# X_t, X_v, y_t, y_v = train_test_split(X_res, y_res, test_size=0.18)
# dtree = DTC45()
# dtree.fit(X_t, y_t, cur_attrs, attr_is_discrete=[x in categorical_attris for x in cur_attrs])
# print('myDtree',calc_metrics(y_test,dtree.predict(X_test)))
# dtree.pruning(X_v, y_v)
# print('After pruning',calc_metrics(y_test,dtree.predict(X_test)))



from classifier.randomforest import RandomForest
myrf = RandomForest(tree_number=30, balance_sample=0)
myrf.fit(X_res, y_res, cur_attrs, attr_is_discrete=[x in categorical_attris for x in cur_attrs])
print('myRF',calc_metrics(y_test,myrf.predict(X_test)))

from classifier.randomforest import RandomForest
myrf_balanced = RandomForest(tree_number=30, balance_sample=1)
myrf_balanced.fit(X_res, y_res, cur_attrs, attr_is_discrete=[x in categorical_attris for x in cur_attrs])
print('myRF',calc_metrics(y_test,myrf_balanced.predict(X_test)))