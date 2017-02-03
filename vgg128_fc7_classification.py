import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
import pandas as pd
from sklearn.linear_model import LogisticRegression

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
            y_pred = clf.predict(X_test)

            y_test_true.append(y_test)
            y_test_pred.append(y_pred)

    y_test_true = np.concatenate(y_test_true)
    y_test_pred = np.concatenate(y_test_pred)

    print classification_report(y_test_true, y_test_pred)
    print 'cross validation is Stratified K Fold with K=', n_folds
    print 'Confusion matrix is:'
    print confusion_matrix(y_test_true, y_test_pred)
    return accuracy_score(y_test_true, y_test_pred)

features = np.loadtxt('vgg128_fc7_feature_batch_1.txt')
# features_norm = normalize(features)
train_y = pd.read_csv('id_train.csv', dtype=object)
labels = train_y.label.values

print '--- start training ---'
# clf_rf = RandomForestClassifier(n_estimators=300, class_weight='balanced')
# print train_folds(clf_rf, features, labels)

clf_svm = SVC(kernel='rbf', C=10, gamma=0.001, decision_function_shape='ovr')
print train_folds(clf_svm, features, labels)

clf_lr = LogisticRegression(penalty='l2', C=0.003, multi_class='ovr', class_weight='balanced')
print train_folds(clf_lr, features, labels)

clf_svm.fit(features, labels)


print '--- prediction ---'
# prediction

df_submission = pd.read_csv('sample_submission4.csv', dtype=object)
features_test = np.loadtxt('vgg128_fc7_feature_batch_1_test.txt')
pred_y = clf_svm.predict(features_test)

df_submission['label'] = pred_y
df_submission.to_csv('submission.csv', index=False)