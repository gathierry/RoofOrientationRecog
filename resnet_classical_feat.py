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


def train_folds(clf, X, y, n_folds=4):
    y_test_true = []
    y_test_pred = []
    for i in xrange(3):
        skf = StratifiedKFold(y, n_folds, random_state=i, shuffle=True)
        for train_index, test_index in skf:
            X_train = X[train_index, :]
            X_test = X[test_index, :]
            y_train = y[train_index]
            y_test = y[test_index]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            # y_pred_prob = clf.predict_proba(X_test)
            # max_prob = np.amax(y_pred_prob, axis=1)
            # doubt_idx = [max_prob < 0.55]
            # # take it as 4
            # y_pred[doubt_idx] = '4'
            # # train with only 3 and 4
            # clf.fit(X_train[(y_train == '3') + (y_train == '4')], y_train[(y_train == '3') + (y_train == '4')])
            # y_pred2 = clf.predict(X_test[doubt_idx])
            # y_pred[doubt_idx] = y_pred2

            #print itemfreq(y_pred)
            y_test_true.append(y_test)
            y_test_pred.append(y_pred)

    y_test_true = np.concatenate(y_test_true)
    y_test_pred = np.concatenate(y_test_pred)

    print classification_report(y_test_true, y_test_pred)
    print 'cross validation is Stratified K Fold with K=', n_folds
    print 'Confusion matrix is:'
    print confusion_matrix(y_test_true, y_test_pred)
    return accuracy_score(y_test_true, y_test_pred)


resnet_feat = np.loadtxt('resnet_feature_batch_1.txt')
# features_norm = normalize(features)
train_y = pd.read_csv('id_train.csv', dtype=object)
labels = train_y.label.values
resnet_feat_test = np.loadtxt('resnet_feature_batch_1_test.txt')
df_submission = pd.read_csv('sample_submission4.csv', dtype=object)

df_old = pd.read_csv('data_images_70_160623.csv',  index_col=0, dtype=object)
df_old = df_old.reindex(np.int32(train_y.Id.values))
old_feat = df_old.values[:8000, :]
old_feat_test = df_old.values[8000:, :]

features = np.concatenate((resnet_feat, old_feat), axis=1)
clf_lr = LogisticRegression(penalty='l2', C=0.003, multi_class='ovr', class_weight='balanced')
print train_folds(clf_lr, old_feat, labels)