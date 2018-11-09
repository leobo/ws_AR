from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
# Imports
import tensorflow as tf
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.model_selection import KFold, cross_val_score

tf.logging.set_verbosity(tf.logging.INFO)


# Our application logic will be added here

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer with dynamic batch size, size 28*28 and only one color channel
    input_layer = tf.reshape(features["x"], [-1, 320, 240, 2])
    # input_layer = features["x"]
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # Dense Layer 1
    pool2_flat = tf.reshape(pool3, [-1, 30 * 40 * 32])
    dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # dropout1 = tf.layers.dropout(
    #     inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Dense Layer 2
    dense2 = tf.layers.dense(inputs=dense1, units=1024, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout2, units=101)
    # Network construction done!

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=101)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.08)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def read_surfWS_tar(path):
    surfWS_path = []
    tar = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        surfWS_path += [os.path.join(dirpath, f) for f in filenames if f != '.DS_Store']
        tar += [f.split()[0].split('_')[1] for f in filenames if f != '.DS_Store']
    t = len(surfWS_path)
    surfWS_des = np.zeros(shape=(t, 240, 320, 2), dtype=np.float32)
    for i in range(t):
        tem = np.load(surfWS_path[i])
        x, y, c = tem.shape
        if x != 240 or y != 320 or c != 2:
            print(surfWS_path[i])
            continue
        surfWS_des[i, :, :, 0] = preprocessing.normalize(tem[:, :, 0], norm='l2')
        surfWS_des[i, :, :, 1] = preprocessing.normalize(tem[:, :, 1], norm='l2')
    return surfWS_des, np.array(tar)


def main():
    # # Load training and eval data
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # des1 = mnist.train.images  # Returns np.array
    # tar = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images  # Returns np.array
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    des, tar = read_surfWS_tar("/home/boy2/UCF101/ws_over_TD_2channels")

    tar = preprocessing.LabelEncoder().fit_transform(tar)


    train_data, eval_data, train_labels, eval_labels = model_selection.train_test_split(des, tar, test_size=0.05,
                                                                                        random_state=42)

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1000)


    # k_fold = KFold(n_splits=3)
    # score = 0
    #for train_indices, test_indices in k_fold.split(des):
        # train_data = des[train_indices]
        # train_labels = tar[train_indices]
        # eval_data = des[test_indices]
        # eval_labels = tar[test_indices]
        # Set up logging for predictions
        # , model_dir="/home/boy2/UCF101/cnn_model"
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=50,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=15000,
        hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    #     score += eval_results['accuracy']
    # print('The score is: ' + str(score/3))


if __name__ == "__main__":
    main()
