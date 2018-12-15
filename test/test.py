from collections import Counter
from math import sqrt

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from classifier.dtree import DTC45
from classifier.nbayes import NBayes
from classifier.randomforest import RandomForest

# function for calculating classification performance metrics
def calc_metrics(y_label, y_pred):
    con_matrix = confusion_matrix(y_label, y_pred)
    TN = float(con_matrix[0][0])
    FP = float(con_matrix[0][1])
    FN = float(con_matrix[1][0])
    TP = float(con_matrix[1][1])
    P = TP + FN
    N = TN + FP
    Sn = TP / P
    Sp = TN / N
    Acc = (TP + TN) / (P + N)
    Pre = (TP) / (TP + FP)
    Recall = (TP) / (TP + FN)
    F1_score = 2 * Pre * Recall / (Pre + Recall)
    MCC = 0
    tmp = sqrt((TP + FP) * (TP + FN)) * sqrt((TN + FP) * (TN + FN))
    if tmp != 0:
        MCC = (TP * TN - FP * FN) / tmp
    return Sn, Sp, Acc, Pre, F1_score, MCC


cur_attrs = ['age', 'job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'campaign',
             'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m',
             'nr.employed']
categorical_attris = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                      'poutcome']
metric_names = ['Sensitivity/Recall', 'Specificity', 'Accuracy', 'Precision', 'F1-Score', 'MCC', 'AUC']


bank = pd.read_csv('../data/bank.csv')
X = np.array(bank.ix[:, cur_attrs])
y = np.array(bank.ix[:, bank.columns[-1]])
X_y = np.hstack([X, y.reshape(len(y), 1)])
np.random.shuffle(X_y)
X, y = X_y[:, 0:-1], X_y[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

print('\n# Smote Re-sampling.')
smote = SMOTE(ratio=0.5, sampling_strategy='minority')
X_res, y_res = smote.fit_sample(X_train, y_train)
X_y_res = np.hstack([X_res, y_res.reshape(len(y_res), 1)])
np.random.shuffle(X_y_res)
X_res, y_res = X_y_res[:, 0:-1], X_y_res[:, -1]
for i, attr in enumerate(cur_attrs):
    if attr in categorical_attris:
        X_res[:, i] = X_res[:, i].astype('int32')
print('Current y_train distribution:\n' + str(Counter(y_res)))

print("\n# Calculating following metrics for each model.\n"+str(metric_names))

print('\n# NBayes')
mynb = NBayes()
mynb.fit(X_res, y_res, cur_attrs, attr_is_discrete=[x in categorical_attris for x in cur_attrs])
print(calc_metrics(y_test, mynb.predict(X_test)))

print('\n# DTree')
mydt = DTC45(max_continuous_attr_splits=50)
mydt.fit(X_res, y_res, cur_attrs, attr_is_discrete=[x in categorical_attris for x in cur_attrs])
print(calc_metrics(y_test, mydt.predict(X_test)))

print('\n# Dtree-Pruning')
X_t, X_v, y_t, y_v = train_test_split(X_res, y_res, test_size=0.18)
dtree = DTC45(max_continuous_attr_splits=50)
dtree.fit(X_t, y_t, cur_attrs, attr_is_discrete=[x in categorical_attris for x in cur_attrs])
print('Before pruning\n', calc_metrics(y_test, dtree.predict(X_test)))
dtree.pruning(X_v, y_v)
print('After pruning\n', calc_metrics(y_test, dtree.predict(X_test)))

print('\n# RF')
print("Random Forest is kind of time consuming, please wait.")
myrf = RandomForest(tree_number=50, balance_sample=0)
myrf.fit(X_res, y_res, cur_attrs, attr_is_discrete=[x in categorical_attris for x in cur_attrs])
print(calc_metrics(y_test, myrf.predict(X_test)))

print('\n# RF_Balanced')
myrf_balanced = RandomForest(tree_number=50, balance_sample=1)
myrf_balanced.fit(X_res, y_res, cur_attrs, attr_is_discrete=[x in categorical_attris for x in cur_attrs])
print(calc_metrics(y_test, myrf_balanced.predict(X_test)))

# Compute ROC curve and AUC
model_name_dict = {'NBayes': mynb, 'RF': myrf, 'RF-Balanced': myrf_balanced}
for model in model_name_dict:
    fpr, tpr, thresholds = roc_curve(y_test, model_name_dict[model].predict_proba(X_test))
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, alpha=0.8, label='ROC of %s (AUC = %0.4f)' % (model, roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, label='Luck', alpha=.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC and AUC comparison', fontsize=16)
plt.legend(loc="lower right")
plt.savefig('../result/ROC-AUC.pdf')
plt.show()
plt.close()

# The performance of Random Forest with different number of trees
rf_metrcis = []
for i in range(50):
    metrics = []
    metrics += calc_metrics(y_test, myrf.predict(X_test, predict_tree_num=i + 1))
    fpr, tpr, thresholds = roc_curve(y_test, myrf.predict_proba(X_test, predict_tree_num=i + 1))
    roc_auc = auc(fpr, tpr)
    metrics += [roc_auc]
    rf_metrcis.append(metrics)
rf_metrcis = np.array(rf_metrcis)
plt.figure()
for i, metric in enumerate(metric_names):
    plt.plot(rf_metrcis[:, i], lw=1.5, label=metric)
plt.xlabel('Number of trees in Random Forest', fontsize=13)
plt.ylim([0.2, 1.36])
plt.title('Performance of Random Forest', fontsize=15)
plt.legend(loc="upper right", fontsize=8.15)
plt.savefig('../result/RF-performances.pdf')
plt.show()
plt.close()

# The performance of Random Forest-Balanced  with different number of trees
rf_metrcis_b = []
for i in range(50):
    metrics = []
    metrics += list(calc_metrics(y_test, myrf_balanced.predict(X_test, predict_tree_num=i + 1)))
    fpr, tpr, thresholds = roc_curve(y_test, myrf_balanced.predict_proba(X_test, predict_tree_num=i + 1))
    roc_auc = auc(fpr, tpr)
    metrics.append(roc_auc)
    rf_metrcis_b.append(metrics)
rf_metrcis_b = np.array(rf_metrcis_b)
plt.figure()
for i, metric in enumerate(metric_names):
    plt.plot(rf_metrcis_b[:, i], lw=1.5, label=metric)
plt.xlabel('Number of trees in Random Forest', fontsize=13)
plt.ylim([0.2, 1.3])
plt.title('Performance of Random Forest-Balanced', fontsize=15)
plt.legend(loc="upper right", fontsize=9)
plt.savefig('../result/RF-Balanced-performances.pdf')
plt.show()
plt.close()
