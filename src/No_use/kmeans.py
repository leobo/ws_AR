from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture


def his_gen(clustering_model, data, NUM_CLUSTERS):
    """
    Return the clustering results.
    :param KMEANS:
    :param data:
    :param NUM_CLUSTERS:
    :return:
    """
    bow = [0] * NUM_CLUSTERS
    labels = clustering_model.predict(data)
    for l in labels:
        bow[l] += 1
    return bow


def kmeans(num, data):
    """
    Create a kmeans classifier with num clusters, than train all the given data and produce the clustering results.
    :param num: number of clusters
    :param data: training data
    :return: the clustering results
    """
    kmeans = MiniBatchKMeans(n_clusters=num, init_size=3 * num)

    data_norm = []
    for d in data:
        data_norm.append(preprocessing.normalize(d))
    data = data_norm

    for d in data:
        kmeans.fit(d)
    bow = []
    for d in data:
        bow.append(his_gen(kmeans, d, num))
    return bow


def GMM_train_gen(num, data):
    gmm = GaussianMixture(n_components=num, covariance_type='tied', init_params='random')

    data_norm = []
    for d in data:
        data_norm.append(preprocessing.normalize(d))
    data = data_norm

    for d in data:
        gmm.fit(d)
    gmm_pred = []
    for d in data:
        gmm_pred.append(his_gen(gmm, d, num))
    return gmm_pred