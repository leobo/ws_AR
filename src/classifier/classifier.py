import numpy
from sklearn import preprocessing, svm, neighbors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.ensemble import BaggingClassifier

def svm_classifier(data, labels):
    """
    Construct and train a linear SVM.
    :param data: input data that is needed to classify
    :param labels: labels for classes
    :return:
    """
    # normalize the data
    data = preprocessing.normalize(data, norm='l2')
    classifier = svm.LinearSVC(penalty='l2', class_weight='balanced',
                               random_state=numpy.random.RandomState(0))
    bagging_classifier = BaggingClassifier(classifier, n_estimators=10, max_samples=1/2, n_jobs=1)
    fold = KFold(n_splits=10)
    predicts = cross_val_predict(bagging_classifier, data, labels, n_jobs=-1, cv=fold)
    acc_score = accuracy_score(labels, predicts)
    return predicts, acc_score


def knn_classifier(data, tar):
    """
    Construct and train a knn classifier.
    :param data: input data that is needed to classify
    :param tar: labels for classes
    :return:
    """
    neig = neighbors.KNeighborsClassifier(weights='distance', n_neighbors=10)
    fold = KFold(n_splits=10)
    data = preprocessing.normalize(data, norm='l2')
    predicts = cross_val_predict(neig, data, tar, n_jobs=-1, cv=fold)
    acc_score = accuracy_score(tar, predicts)
    return predicts, acc_score
