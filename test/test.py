from math import sqrt
from sklearn.metrics import roc_curve, auc
# from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from classifier.dtree import DTC45
from classifier.nbayes import NBayes
from classifier.randomforest import RandomForest


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
    Pre = (TP) / (TP + FP)
    Recall = (TP) / (TP + FN)
    F1_score = 2*Pre*Recall/(Pre+Recall)
    MCC = 0
    tmp = sqrt((TP + FP) * (TP + FN)) * sqrt((TN + FP) * (TN + FN))
    if tmp != 0:
        MCC = (TP * TN - FP * FN) / tmp
    fpr, tpr, thresholds = roc_curve(y_label, y_proba)
    AUC = auc(fpr, tpr)
    return Sn, Sp, Acc, Pre, F1_score, MCC, AUC

cur_attrs = ['age', 'job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'campaign',
             'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m',
             'nr.employed']
categorical_attris =  ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

bank = pd.read_csv('../data/bank.csv')
X = np.array(bank.ix[:, cur_attrs])
y = np.array(bank.ix[:, bank.columns[-1]])
X_y = np.hstack([X, y.reshape(len(y),1)])
np.random.shuffle(X_y)
X, y = X_y[:,0:-1], X_y[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# rf = RandomForestClassifier(50)
# rf.fit(X_train, y_train)
# print('RF',calc_metrics(y_test,rf.predict(X_test)))
#
# nb = GaussianNB()
# nb.fit(X_train, y_train)
# print('NBayes',calc_metrics(y_test,nb.predict(X_test)))
#
# lr = LogisticRegression(max_iter=500)
# lr.fit(X_train, y_train)
# print('LR',calc_metrics(y_test,lr.predict(X_test)))
#
#
# mynb = NBayes()
# mynb.fit(X_train, y_train, cur_attrs, attr_is_discrete=[x in categorical_attris for x in cur_attrs])
# print('myNBayes',calc_metrics(y_test,mynb.predict(X_test)))

X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.18)
dtree = DTC45(max_continuous_attr_splits=50)
dtree.fit(X_t, y_t, cur_attrs, attr_is_discrete=[x in categorical_attris for x in cur_attrs])
print('myDtree-before-pruning',calc_metrics(y_test,dtree.predict(X_test)))
dtree.pruning(X_v, y_v)
print('After pruning',calc_metrics(y_test,dtree.predict(X_test)))


print('#\n\n Smote Resampling.')
smote = SMOTE(ratio=0.5, sampling_strategy='minority')
X_res, y_res = smote.fit_sample(X_train, y_train)
X_y_res = np.hstack([X_res, y_res.reshape(len(y_res),1)])
np.random.shuffle(X_y_res)
X_res, y_res = X_y_res[:,0:-1], X_y_res[:,-1]
for i, attr in enumerate(cur_attrs):
    if attr in categorical_attris:
        X_res[:, i] = X_res[:,i].astype('int32')
print(Counter(y_res))

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
# #
mynb = NBayes()
mynb.fit(X_res, y_res, cur_attrs, attr_is_discrete=[x in categorical_attris for x in cur_attrs])
print('myNBayes',calc_metrics(y_test,mynb.predict(X_test)))


mydt = DTC45(max_continuous_attr_splits=50)
mydt.fit(X_res, y_res, cur_attrs, attr_is_discrete=[x in categorical_attris for x in cur_attrs])
print('myDTree',calc_metrics(y_test,mydt.predict(X_test)))


X_t, X_v, y_t, y_v = train_test_split(X_res, y_res, test_size=0.18)
dtree = DTC45(max_continuous_attr_splits=50)
dtree.fit(X_t, y_t, cur_attrs, attr_is_discrete=[x in categorical_attris for x in cur_attrs])
print('myDtree-before-pruning',calc_metrics(y_test,dtree.predict(X_test)))
dtree.pruning(X_v, y_v)
print('After pruning',calc_metrics(y_test,dtree.predict(X_test)))


myrf = RandomForest(tree_number=50, balance_sample=0)
myrf.fit(X_res, y_res, cur_attrs, attr_is_discrete=[x in categorical_attris for x in cur_attrs])
print('myRF',calc_metrics(y_test,myrf.predict(X_test)))

myrf_balanced = RandomForest(tree_number=50, balance_sample=1)
myrf_balanced.fit(X_res, y_res, cur_attrs, attr_is_discrete=[x in categorical_attris for x in cur_attrs])
print('myRF_balanced',calc_metrics(y_test,myrf_balanced.predict(X_test)))

# Compute ROC curve and AUC
import matplotlib.pyplot as plt
model_name_dict = {'NBayes': mynb,'RF':myrf,'RF-Balanced':myrf_balanced}
for model in model_name_dict:
    fpr, tpr, thresholds = roc_curve(y_test, model_name_dict[model].predict_proba(X_test))
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, alpha=0.8, label='ROC of %s (AUC = %0.4f)' % (model, roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, label='Luck', alpha=.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC and AUC comparison', fontsize=18)
plt.legend(loc="lower right")
plt.savefig('../result/ROC-AUC.pdf')
plt.show()
plt.close()


metrics = ['Sensitivity', 'Specificity/Recall', 'Accuracy', 'Precision', 'F1-Score', 'MCC', 'AUC']
rf_metrcis = []
for i in range(50):
    rf_metrcis.append(calc_metrics(y_test,myrf.predict(X_test)))
rf_metrcis = np.array(rf_metrcis)
for i,metric in enumerate(metrics):
    plt.plot(rf_metrcis[:,i], lw=1, label=metric)
plt.xlabel('Number of trees in Random Forest', fontsize=16)
# plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC and AUC comparison', fontsize=18)
plt.legend(loc="lower right")
plt.savefig('../result/ROC-AUC.pdf')
plt.show()
plt.close()