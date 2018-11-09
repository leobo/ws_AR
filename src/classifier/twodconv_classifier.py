import tensorflow as tf
from tensorflow.python.ops import init_ops

tf.logging.set_verbosity(tf.logging.INFO)

start_learning_rate = 0.001
# 8 - 64
batch_size = 64
dropout_rate = 0.9
alpha = 0.2
num_unique_classes = 101
test_data_ratio = 0.3
epech_decay = 5
decay_rate = 0.9
epech = 90


def nn_classifier(features, labels, mode, params):
    input_layer = tf.reshape(features['x'],
                             [-1, 64, 64, 3])
    input_layer = tf.cast(input_layer, tf.float32)

    # conv1 = tf.layers.conv2d(inputs=input_layer, filters=16, kernel_initializer=tf.contrib.layers.xavier_initializer(),
    #                          bias_initializer=tf.contrib.layers.xavier_initializer(), kernel_size=[3, 3], padding="same",
    #                          activation=None)
    # bn1 = tf.layers.batch_normalization(conv1, axis=-1, center=True, scale=True,
    #                                     training=(mode == tf.estimator.ModeKeys.TRAIN))
    # lrelu1 = tf.nn.relu(conv1)
    # pool1 = tf.layers.max_pooling2d(inputs=lrelu1, pool_size=[2, 2], strides=2)

    # conv2 = tf.layers.conv2d(inputs=input_layer, filters=16, kernel_initializer=tf.contrib.layers.xavier_initializer(),
    #                          bias_initializer=tf.constant_initializer(0.1), kernel_size=[3, 3], padding="same",
    #                          activation=None)
    # bn2 = tf.layers.batch_normalization(conv2, axis=-1, center=True, scale=True,
    #                                     training=(mode == tf.estimator.ModeKeys.TRAIN))
    # lrelu2 = tf.nn.relu(conv2)
    # pool2 = tf.layers.max_pooling2d(inputs=lrelu2, pool_size=[2, 2], strides=2)
    #
    conv3 = tf.layers.conv2d(inputs=input_layer, filters=128, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=tf.constant_initializer(0.1), kernel_size=[3, 3], padding="same",
                             activation=tf.nn.tanh)
    # bn3 = tf.layers.batch_normalization(conv3, axis=-1, center=True, scale=True,
    #                                     training=(mode == tf.estimator.ModeKeys.TRAIN))
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    conv4 = tf.layers.conv2d(inputs=pool3, filters=128, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=tf.constant_initializer(0.1), kernel_size=[3, 3], padding="same",
                             activation=tf.nn.tanh)
    # bn4 = tf.layers.batch_normalization(conv4, axis=-1, center=True, scale=True,
    #                                     training=(mode == tf.estimator.ModeKeys.TRAIN))
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    pool3_flat = tf.reshape(pool4, [-1, 16*16*128])

    # fc1 = tf.layers.dense(inputs=pool3_flat, units=4096, bias_initializer=tf.constant_initializer(0.1),
    #                       activation=None)
    # bnfc1 = tf.layers.batch_normalization(fc1, axis=-1, center=True, scale=True,
    #                                       training=(mode == tf.estimator.ModeKeys.TRAIN))
    # lrelufc1 = tf.nn.relu(bnfc1)
    #
    fc2 = tf.layers.dense(inputs=pool3_flat, units=2048, bias_initializer=tf.constant_initializer(0.1),
                          activation=tf.nn.tanh)
    # bnfc2 = tf.layers.batch_normalization(fc2, axis=-1, center=True, scale=True,
    #                                       training=(mode == tf.estimator.ModeKeys.TRAIN))
    dropoutfc2 = tf.layers.dropout(
        inputs=fc2, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN))

    fc3 = tf.layers.dense(inputs=dropoutfc2, units=2048, bias_initializer=tf.constant_initializer(0.1),
                          activation=tf.nn.tanh)
    # bnfc3 = tf.layers.batch_normalization(fc3, axis=-1, center=True, scale=True,
    #                                       training=(mode == tf.estimator.ModeKeys.TRAIN))

    dropoutfc3 = tf.layers.dropout(
        inputs=fc3, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Logits Layer
    logits = tf.layers.dense(inputs=dropoutfc3, units=num_unique_classes)

    # if mode == tf.estimator.ModeKeys.PREDICT:
    #     return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=num_unique_classes)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    learning_rate = tf.train.exponential_decay(
        start_learning_rate,  # Base learning rate.
        tf.train.get_global_step() * batch_size,  # Current index into the dataset.
        params['train_set_size'] * epech_decay,  # Decay step.
        decay_rate,  # Decay rate.
        staircase=True)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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


def classify(train_data, train_labels, eval_data, eval_labels):
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)
    params = {"train_set_size": len(train_data), "feature_len": len(train_data[0])}
    mnist_classifier = tf.estimator.Estimator(model_fn=nn_classifier, params=params)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=batch_size,
        num_epochs=epech,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=None,
        hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        batch_size=batch_size,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    return eval_results
