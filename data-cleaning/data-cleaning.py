import math
import operator
import os
import random
import time
import sys
import bisect
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from minepy import MINE
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from classifier.dtree import DTC45

source_data_dir = r"../data/source-data/bank-additional-full.csv"
# source_data_dir = r"../data/source-data/bank-additional-full.csv"

#=====================================================
# load data

df = pd.read_csv(filepath_or_buffer=source_data_dir,sep=';')
# attribute 'duration' should be discarded
df = df.drop('duration', axis=1)
print(df.info())
print(df.describe())

tot_attrs = list(df.columns)
numeric_attris = list(df.describe().columns)
categorical_attris = [x for x in tot_attrs if x not in numeric_attris]
#=====================================================



#=====================================================
# deal with unknown values by prediction

tot_records_num = df.shape[0]
attrs_have_unknown = []
print("# Number of unknown values in each categorical attribute:")
for attr in categorical_attris:
    number_unknown = len([x for x in df[attr] if x=='unknown'])
    if number_unknown>0:
        attrs_have_unknown.append(attr)
    print('Attribute: %s, unknown values: %d(%.3f%%).' % (attr, number_unknown, number_unknown*1.0/tot_records_num*100))
attrs_dont_have_unknown = [x for x in tot_attrs if x not in attrs_have_unknown]

df_columns_have_known = df.ix[:, attrs_have_unknown]
tot_unknown_record_indexes = [i for i in df.index if 'unknown' in list(df_columns_have_known.ix[i,:])]
train_record_indexes = [i for i in df.index if i not in tot_unknown_record_indexes]
train_records = df.ix[train_record_indexes, :]

for attr in attrs_have_unknown:
    print("\nPredicting unknown values in attribute %s..." % attr)
    tree = DTC45(max_depth=40, min_samples_split=3, max_continuous_attr_splits=200)
    attr_list = attrs_dont_have_unknown

    print("Training...")
    tree.fit(X_train=np.array(train_records.ix[:,attr_list]), y_train=train_records[attr].values, attr_list=attr_list,
             attr_is_discrete=[x in categorical_attris for x in attr_list], verbose=0)
    print("Overall Accuracy on train data: "+str(tree.evaluate(train_records.ix[:,attr_list].values,train_records[attr].values)))

    test_indexes = [x for x in tot_unknown_record_indexes if df.ix[x, attr]=='unknown']
    print("Predicting...")
    df.ix[test_indexes, attr] = tree.predict(np.array(df.ix[test_indexes, attr_list]))
#=====================================================



#=====================================================
# code each categorical attribute

job_values = ["entrepreneur","admin.","management","blue-collar","technician","self-employed","services","housemaid","retired","student","unemployed"]
marital_values = ["divorced","married","single"]
education_values = ["illiterate","basic.4y","basic.6y","basic.9y","high.school","professional.course","university.degree"]
default_values = ["no","yes"]
housing_values = ["no","yes"]
loan_values = ["no","yes"]
contact_values = ["cellular","telephone"]
month_values = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
day_of_week_values = ["mon","tue","wed","thu","fri"]
poutcome_values = ["nonexistent","failure","success"]
y_values = ["yes","no"]

job_coding = dict(zip(job_values,list(range(len(job_values)))))
marital_coding = dict(zip(marital_values,list(range(len(marital_values)))))
education_coding = dict(zip(education_values,list(range(len(education_values)))))
default_coding = dict(zip(default_values,list(range(len(default_values)))))
housing_coding = dict(zip(housing_values,list(range(len(housing_values)))))
loan_coding = dict(zip(loan_values,list(range(len(loan_values)))))
contact_coding = dict(zip(contact_values,list(range(len(contact_values)))))
month_coding = dict(zip(month_values,list(range(len(month_values)))))
day_of_week_coding = dict(zip(day_of_week_values,list(range(len(day_of_week_values)))))
poutcome_coding = dict(zip(poutcome_values,list(range(len(poutcome_values)))))
y_coding = dict(zip(y_values,list(range(len(y_values)))))

attrlabel2value_map = {
'job':job_coding, 'marital':marital_coding, 'education':education_coding, 'default':default_coding, 'housing':housing_coding,
'loan':loan_coding, 'contact':contact_coding, 'month':month_coding, 'day_of_week':day_of_week_coding, 'poutcome':poutcome_coding,
'y':y_coding
}


for attr in attrlabel2value_map.keys():
    attr_map = attrlabel2value_map[attr]
    df.ix[: ,attr] = [attr_map[x] for x in df.ix[: ,attr]]
#=====================================================



#=====================================================
# 特征筛选
from minepy import MINE
from scipy import stats

attrs = tot_attrs[0:-1]
print("# Feature ranking process.")

# t-test based feature ranking attribute
pos = [i for i in df.index if df.ix[i,'y']==1]
neg = [i for i in df.index if df.ix[i,'y']==0]
t_vals, p_vals = stats.ttest_ind(df.ix[pos, attrs], df.ix[neg, attrs], axis=0)
ttest_result = sorted(list(zip(attrs, p_vals)), key=operator.itemgetter(1))
ttest_sorted_attrs = [x[0] for x in ttest_result]
ttest_ranking = [ttest_sorted_attrs.index(x)+1 for x in attrs]
print(ttest_sorted_attrs)
print(ttest_ranking)

# MIC (Maximal Information Coefficient) based feature ranking attribute
mine = MINE()
mic_scores = []
y_values = df.ix[:,'y']
for attr in tot_attrs[0:-1]:
    mine.compute_score(df.ix[:, attr], y_values)
    mic_scores.append(mine.mic())
mic_result = sorted(list(zip(attrs, mic_scores)), key=operator.itemgetter(1), reverse=True)
mic_sorted_attrs = [x[0] for x in mic_result]
mic_ranking = [mic_sorted_attrs.index(x)+1 for x in attrs]
print(mic_sorted_attrs)
print(mic_ranking)

# RFE (Recursive Feature Elimination) feature ranking
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

estimator = SVR(kernel="linear")
selector = RFE(estimator, 1, step=1)
selector = selector.fit(df.ix[:, attrs], y_values)
rfe_ranking = selector.ranking_

print(rfe_ranking)


# RF (Random Forest) feature ranking
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50)
rf.fit(df.ix[:, attrs], y_values)
rf_scores = rf.feature_importances_
print(rf_scores)
rf_result = sorted(list(zip(attrs, rf_scores)), key=operator.itemgetter(1), reverse=True)
rf_sorted_attrs = [x[0] for x in rf_result]
rf_ranking = [rf_sorted_attrs.index(x)+1 for x in attrs]
print(rf_sorted_attrs)
print(rf_ranking)


# draw the violin plot for each attribute
fig_rows = 4
fig_columns = 5
fig, axes = plt.subplots(fig_rows, fig_columns, figsize=(20,15), sharex=True)
sns.set(font_scale=2)
for i in range(fig_rows):
    for j in range(fig_columns):
        fig_num = i*fig_columns + (j+1)
        attr = tot_attrs[fig_num-1]
        print(fig_num)
        sns.violinplot(y=list(df[attr]), x=list(df["y"]) if i<3 else df["y"], ax=axes[i, j], annot_kws={"size": 18}).set_title(attr)
        plt.title(attr)
plt.tight_layout(w_pad=0.15,h_pad=0.05)
plt.savefig('../result/attr_distribution.pdf')
plt.show()
plt.close()

#=====================================================

# 标准化

# 类别不平衡处理
from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_res, y_res = smote_enn.fit_sample(np.array(df.ix[:, attrs]), y_values)


df.to_csv('../data/bank.csv')