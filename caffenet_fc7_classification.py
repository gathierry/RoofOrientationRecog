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

def normalize(x):
    min_x = np.amin(x, axis=0)
    max_x = np.amax(x, axis=0)
    return (x - min_x) / (max_x - min_x)


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
            y_pred = clf_lr.predict(X_test)
            y_pred_prob = clf_lr.predict_proba(X_test)
            max_prob = np.amax(y_pred_prob, axis=1)
            doubt_idx = [max_prob < 0.4]
            # train with only 3 and 4
            clf_lr.fit(X_train[(y_train == '3') + (y_train == '4')], y_train[(y_train == '3') + (y_train == '4')])
            y_pred2 = clf_lr.predict(X_test[doubt_idx])
            y_pred[doubt_idx] = y_pred2

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

features = np.loadtxt('caffenet_fc7_feature_batch_1.txt')
# features_norm = normalize(features)
train_y = pd.read_csv('id_train.csv', dtype=object)
labels = train_y.label.values
features_test = np.loadtxt('caffenet_fc7_feature_batch_1_test.txt')
df_submission = pd.read_csv('sample_submission4.csv', dtype=object)



# all_x = np.concatenate((features, features_test))
#
# pca = PCA()
# pca.fit(all_x)
# sum(pca.explained_variance_ratio_[:2700])
# plt.plot(pca.explained_variance_ratio_, linewidth=2)
#
# pca = PCA(n_components=2700)
# X_r = pca.fit(all_x).transform(all_x)
#
# np.savetxt('caffenet_fc7_feature_batch_1.txt', X_r[:8000, :])
# np.savetxt('caffenet_fc7_feature_batch_1_test.txt', X_r[8000:, :])

# features = np.loadtxt('caffenet_fc7_feature_batch_1.txt')
# features_test = np.loadtxt('caffenet_fc7_feature_batch_1_test.txt')
print '--- start training ---'
# clf_rf = RandomForestClassifier(n_estimators=300, class_weight='balanced')
# print train_folds(clf_rf, features, labels)
#
# clf_svm = SVC(kernel='rbf', C=3, gamma=0.001, decision_function_shape='ovr', class_weight='balanced')
# print train_folds(clf_svm, features, labels)

clf_lr = LogisticRegression(C=0.01, multi_class='ovr', class_weight='balanced')
print train_folds(clf_lr, features, labels)


clf_lr.fit(features, labels)


print '--- prediction ---'
pred_y = clf_lr.predict(features_test)
y_pred_prob = clf_lr.predict_proba(features_test)
max_prob = np.amax(y_pred_prob, axis=1)
doubt_idx = [max_prob < 0.45]
# train with only 3 and 4
clf_lr.fit(features[(labels == '3') + (labels == '4')], labels[(labels == '3') + (labels == '4')])
pred_y2 = clf_lr.predict(features_test[doubt_idx])
pred_y[doubt_idx] = pred_y2


print itemfreq(pred_y)
df_submission['label'] = pred_y
df_submission.to_csv('submission.csv', index=False)