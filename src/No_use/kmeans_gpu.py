import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans


def histograming(list_data, array_labels, num_clusters):
    bow = []
    for ld in list_data:
        row, _ = np.shape(ld)
        single_his = [0] * num_clusters
        for ad in array_labels[0:row - 1]:
            single_his[ad] += 1
        array_labels = np.delete(array_labels, range(row - 1))
        bow.append(single_his)
    return bow


def kmeans(num_clusters, data):
    num_steps = 100
    data_array = np.concatenate(data, axis=0)
    row, col = np.shape(data_array)
    # Input features
    X = tf.placeholder(tf.float32, shape=[row, col])

    # Define the Kmeans
    kmeans = KMeans(inputs=X, num_clusters=num_clusters, distance_metric='squared_euclidean',
                    initial_clusters='kmeans_plus_plus', use_mini_batch=True)

    all_scores, cluster_idx, scores, cluster_centers_initialized, cluster_centers_var, init_op, training_op = \
        kmeans.training_graph()
    # fix for cluster_idx being a tuple
    cluster_idx = cluster_idx[0]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer(), feed_dict={X: data_array})
        sess.run(init_op, feed_dict={X: data_array})

        for i in range(num_steps):
            _, idx = sess.run([training_op, cluster_idx], feed_dict={X: data_array})

    # Create the k means histogram
    return histograming(data, idx, num_clusters)
