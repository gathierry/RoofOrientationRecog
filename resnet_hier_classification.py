import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import itemfreq
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_validation import train_test_split


features = np.loadtxt('resnet_feature_batch_1.txt')
# features_norm = normalize(features)
train_y = pd.read_csv('id_train.csv', dtype=object)
labels = train_y.label.values
features_test = np.loadtxt('resnet_feature_batch_1_test.txt')
df_submission = pd.read_csv('sample_submission4.csv', dtype=object)

clf_svm = SVC(kernel='rbf', C=3, gamma=0.001, decision_function_shape='ovr', class_weight='balanced', probability=True)
clf_lr = LogisticRegression(penalty='l2', C=0.03, multi_class='ovr', class_weight='balanced')
clf_lr2 = LogisticRegression(penalty='l2', C=0.03, multi_class='ovr', class_weight='balanced')
clf_lr3 = LogisticRegression(penalty='l2', C=0.03, multi_class='ovr', class_weight='balanced')
clf_gbdt = GradientBoostingClassifier(n_estimators=500, max_features='sqrt')


eclf = VotingClassifier(estimators=[('lr', clf_lr), ('gbdt', clf_gbdt), ('svm', clf_svm)], voting='soft')
# ------ all --------
X = features
y = labels
y2 = labels.copy()
y2[y2=='1'] = '5'
y2[y2=='2'] = '5'
y2[y2=='3'] = '6'
y2[y2=='4'] = '6'
clf1 = clf_lr
clf2 = clf_lr2
clf3 = clf_lr3
y_test_true = []
y_test_pred = []
for i in xrange(3):
    skf = StratifiedKFold(y, 4, random_state=i, shuffle=True)
    for train_index, test_index in skf:
        X_train = X[train_index, :]
        X_test = X[test_index, :]
        y_train = y[train_index]
        y_test = y[test_index]
        y2_train = y2[train_index]
        y2_test = y2[test_index]
        clf1.fit(X_train, y2_train)
        clf2.fit(X_train[(y_train=='1') | (y_train=='2')], y_train[(y_train=='1') | (y_train=='2')])
        clf3.fit(X_train[(y_train=='3') | (y_train=='4')], y_train[(y_train=='3') | (y_train=='4')])
        y_pred = clf1.predict(X_test)
        y_pred[y_pred=='5'] = clf2.predict(X_test[y_pred=='5'])
        y_pred[y_pred=='6'] = clf3.predict(X_test[y_pred=='6'])
        y_test_true.append(y_test)
        y_test_pred.append(y_pred)

y_test_true = np.concatenate(y_test_true)
y_test_pred = np.concatenate(y_test_pred)

print classification_report(y_test_true, y_test_pred)
print 'cross validation is Stratified K Fold with K=', 4
print 'Confusion matrix is:'
C = confusion_matrix(y_test_true, y_test_pred)
print C
print accuracy_score(y_test_true, y_test_pred)

# -------- 1 and 2 ----------
X = features[(labels=='1') | (labels=='2')]
y = labels[(labels=='1') | (labels=='2')]
clf = clf_svm
y_test_true = []
y_test_pred = []
for i in xrange(3):
    skf = StratifiedKFold(y, 4, random_state=i, shuffle=True)
    for train_index, test_index in skf:
        X_train = X[train_index, :]
        X_test = X[test_index, :]
        y_train = y[train_index]
        y_test = y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_test_true.append(y_test)
        y_test_pred.append(y_pred)

y_test_true = np.concatenate(y_test_true)
y_test_pred = np.concatenate(y_test_pred)

print classification_report(y_test_true, y_test_pred)
print 'cross validation is Stratified K Fold with K=', 4
print 'Confusion matrix is:'
C = confusion_matrix(y_test_true, y_test_pred)
print C
print accuracy_score(y_test_true, y_test_pred)

# --------- 3 and 4 -----------------
X = features[(labels=='3') | (labels=='4')]
y = labels[(labels=='3') | (labels=='4')]
clf = clf_lr
y_test_true = []
y_test_pred = []
for i in xrange(3):
    skf = StratifiedKFold(y, 4, random_state=i, shuffle=True)
    for train_index, test_index in skf:
        X_train = X[train_index, :]
        X_test = X[test_index, :]
        y_train = y[train_index]
        y_test = y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_test_true.append(y_test)
        y_test_pred.append(y_pred)

y_test_true = np.concatenate(y_test_true)
y_test_pred = np.concatenate(y_test_pred)

print classification_report(y_test_true, y_test_pred)
print 'cross validation is Stratified K Fold with K=', 4
print 'Confusion matrix is:'
C = confusion_matrix(y_test_true, y_test_pred)
print C
print accuracy_score(y_test_true, y_test_pred)

# ------- 12 and 34 -----------
X = features
y = labels
y2 = labels.copy()
y2[y2=='1'] = '5'
y2[y2=='2'] = '5'
y2[y2=='3'] = '6'
y2[y2=='4'] = '6'
clf = clf_svm
y_test_true = []
y_test_pred = []
for i in xrange(3):
    skf = StratifiedKFold(y, 4, random_state=i, shuffle=True)
    for train_index, test_index in skf:
        X_train = X[train_index, :]
        X_test = X[test_index, :]
        y_train = y[train_index]
        y_test = y[test_index]
        y2_train = y2[train_index]
        y2_test = y2[test_index]
        clf.fit(X_train, y2_train)
        y_pred = clf.predict(X_test)
        y_test_true.append(y2_test)
        y_test_pred.append(y_pred)

y_test_true = np.concatenate(y_test_true)
y_test_pred = np.concatenate(y_test_pred)

print classification_report(y_test_true, y_test_pred)
print 'cross validation is Stratified K Fold with K=', 4
print 'Confusion matrix is:'
C = confusion_matrix(y_test_true, y_test_pred)
print C
print accuracy_score(y_test_true, y_test_pred)
