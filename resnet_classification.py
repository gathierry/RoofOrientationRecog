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

            print itemfreq(y_pred)
            y_test_true.append(y_test)
            y_test_pred.append(y_pred)

    y_test_true = np.concatenate(y_test_true)
    y_test_pred = np.concatenate(y_test_pred)

    print classification_report(y_test_true, y_test_pred)
    print 'cross validation is Stratified K Fold with K=', n_folds
    print 'Confusion matrix is:'
    print confusion_matrix(y_test_true, y_test_pred)
    return accuracy_score(y_test_true, y_test_pred)


# def stack_train_test(clf_list, clf2, X, y, n_folds=4):
#     y_test_true = []
#     y_test_pred = []
#     for i in xrange(3):
#         skf = StratifiedKFold(y, n_folds, random_state=i, shuffle=True)
#         X_add = np.zeros((X.shape[0], 4 * len(clf_list)))
#         for train_index, test_index in skf:
#             X_train = X[train_index, :]
#             X_test = X[test_index, :]
#             y_train = y[train_index]
#             for k, clf in enumerate(clf_list):
#                 clf.fit(X_train, y_train)
#                 y_pred_prob = clf.predict_proba(X_test)
#                 X_add[test_index, 4*k:4*k+4] = y_pred_prob
#         for train_index, test_index in skf:
#             X_add_train = X_add[train_index, :]
#             X_add_test = X_add[test_index, :]
#             y_train = y[train_index]
#             y_test = y[test_index]
#             clf2.fit(X_add_train, y_train)
#             y_pred = clf2.predict(X_add_test)
#             # print itemfreq(y_pred)
#             y_test_true.append(y_test)
#             y_test_pred.append(y_pred)
#
#     y_test_true = np.concatenate(y_test_true)
#     y_test_pred = np.concatenate(y_test_pred)
#
#     print classification_report(y_test_true, y_test_pred)
#     print 'cross validation is Stratified K Fold with K=', n_folds
#     print 'Confusion matrix is:'
#     print confusion_matrix(y_test_true, y_test_pred)
#     return accuracy_score(y_test_true, y_test_pred)

def train_with_other(clf, X, y, Xoth, yoth, n_folds=4):
    y_test_true = []
    y_test_pred = []
    for i in xrange(3):
        skf = StratifiedKFold(y, n_folds, random_state=i, shuffle=True)
        for train_index, test_index in skf:
            X_train = X[train_index, :]
            X_test = X[test_index, :]
            y_train = y[train_index]
            y_test = y[test_index]
            X_train = np.concatenate((X_train, Xoth), axis=0)
            y_train = np.concatenate((y_train, yoth), axis=0)
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

            print itemfreq(y_pred)
            y_test_true.append(y_test)
            y_test_pred.append(y_pred)

    y_test_true = np.concatenate(y_test_true)
    y_test_pred = np.concatenate(y_test_pred)

    print classification_report(y_test_true, y_test_pred)
    print 'cross validation is Stratified K Fold with K=', n_folds
    print 'Confusion matrix is:'
    print confusion_matrix(y_test_true, y_test_pred)
    return accuracy_score(y_test_true, y_test_pred)


features152 = np.loadtxt('resnet_feature_batch_1.txt')
features101 = np.loadtxt('resnet101_feature_batch_1.txt')
features50 = np.loadtxt('resnet50_feature_batch_1.txt')
features152_test = np.loadtxt('resnet_feature_batch_1_test.txt')
features101_test = np.loadtxt('resnet101_feature_batch_1_test.txt')
features50_test = np.loadtxt('resnet50_feature_batch_1_test.txt')

train_y = pd.read_csv('id_train.csv', dtype=object)
labels = train_y.label.values
df_submission = pd.read_csv('sample_submission4.csv', dtype=object)

# other_features = np.loadtxt('resnet_feature_batch_1_other.txt')
# other_label = np.loadtxt('other_label.txt', dtype=object)[:, 1]


all_x = np.concatenate((np.concatenate((features152, features101, features50), axis=1),
                        np.concatenate((features152_test, features101_test, features50_test), axis=1)), axis=0)

pca = PCA(n_components=2500)
X_r = pca.fit(all_x).transform(all_x)

pca_features = X_r[:8000]
pca_features_test = X_r[8000:]


print '--- start training ---'

clf_gbdt = GradientBoostingClassifier(n_estimators=500, max_features='sqrt')
print train_folds(clf_gbdt, features, labels)

clf_rf = RandomForestClassifier(n_estimators=500)
print train_folds(clf_rf, features, labels)

clf_svm = SVC(kernel='rbf', C=3, gamma=0.001, decision_function_shape='ovr', class_weight='balanced', probability=True)
print train_folds(clf_svm, features, labels)

clf_slin = SVC(kernel='linear', C=0.003, decision_function_shape='ovr', class_weight='balanced', probability=True)
print train_folds(clf_slin, features, labels)

clf_lr = LogisticRegression(penalty='l2', C=0.01, multi_class='ovr', class_weight='balanced')
print train_folds(clf_lr, features, labels)

clf_lda = LinearDiscriminantAnalysis()
print train_folds(clf_lda, features, labels)

eclf1 = VotingClassifier(estimators=[('lr', clf_lr), ('svm', clf_svm)], voting='soft')
print train_folds(eclf1, features, labels)

clf_lr.fit(all_x[:8000, :], labels)
coeff = clf_lr.coef_
#features[:, (np.abs(np.max(coeff, axis=0)) > 0.02)]

features = all_x[:8000, np.abs(np.max(coeff, axis=0)) > 0.02]
features_test = all_x[8000:, np.abs(np.max(coeff, axis=0)) > 0.02]

print '--- prediction ---'
classifier = eclf1
classifier.fit(features, labels)
pred_y = classifier.predict(features_test)

y_pred_prob = classifier.predict_proba(features_test)
max_prob = np.amax(y_pred_prob, axis=1)
doubt_idx = [max_prob < 0.4]
# train with only 3 and 4
classifier.fit(features[(labels == '3') + (labels == '4')], labels[(labels == '3') + (labels == '4')])
pred_y2 = classifier.predict(features_test[doubt_idx])
pred_y[doubt_idx] = pred_y2

print itemfreq(pred_y)
df_submission['label'] = pred_y
df_submission.to_csv('submission.csv', index=False)
