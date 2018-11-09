import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

start_learning_rate = 0.0005
# 8 - 64
batch_size = 64
dropout_rate = 0.9
alpha = 0.2
num_unique_classes = 101
test_data_ratio = 0.3
epech_decay = 3
decay_rate = 0.9
epech = 90


def nn_classifier(features, labels, mode, params):
    input_layer = tf.reshape(features['x'], [-1, 2048*3])
    input_layer = tf.cast(input_layer, tf.float32)
    # dense1 = tf.layers.dense(inputs=input_layer, units=8000, activation=tf.nn.relu)
    # dropout1 = tf.layers.dropout(
    #     inputs=dense1, rate=0.8, training=mode == tf.estimator.ModeKeys.TRAIN)

    # fcn1 = tf.layers.dense(inputs=input_layer, units=8192, bias_initializer=tf.contrib.layers.xavier_initializer(),
    #                        activation=None)
    # relun1 = tf.nn.leaky_relu(fcn1, alpha=alpha)
    #
    # dropoutn1 = tf.layers.dropout(
    #     inputs=relun1, rate=dropout_rate, training=mode == tf.estimator.ModeKeys.TRAIN)
    #
    # # Dense Layer 2
    # fc0 = tf.layers.dense(inputs=input_layer, units=4096, bias_initializer=tf.contrib.layers.xavier_initializer(), activation=None)
    # relu0 = tf.nn.leaky_relu(fc0, alpha=alpha)

    # bn0 = tf.layers.batch_normalization(relu0, axis=1, center=True, scale=True,
    #                                     training=mode == tf.estimator.ModeKeys.TRAIN)
    # dropout0 = tf.layers.dropout(
    #     inputs=relu0, rate=dropout_rate, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Dense Layer 3
    fc1 = tf.layers.dense(inputs=input_layer, units=4096, bias_initializer=tf.contrib.layers.xavier_initializer(),
                          activation=None)
    relu1 = tf.nn.leaky_relu(fc1, alpha=alpha)

    # bn1 = tf.layers.batch_normalization(relu1, axis=1, center=True, scale=True,
    #                                     training=mode == tf.estimator.ModeKeys.TRAIN)
    # dropout1 = tf.layers.dropout(
    #     inputs=bn0, rate=0.8, training=mode == tf.estimator.ModeKeys.TRAIN)
    #
    fc2 = tf.layers.dense(inputs=relu1, units=4096, bias_initializer=tf.contrib.layers.xavier_initializer(),
                          activation=None)
    relu2 = tf.nn.leaky_relu(fc2, alpha=alpha)

    # bn2 = tf.layers.batch_normalization(relu2, axis=1, center=True, scale=True,
    #                                     training=mode == tf.estimator.ModeKeys.TRAIN)
    dropout2 = tf.layers.dropout(
        inputs=relu2, rate=dropout_rate, training=mode == tf.estimator.ModeKeys.TRAIN)

    fc3 = tf.layers.dense(inputs=dropout2, units=2048, bias_initializer=tf.contrib.layers.xavier_initializer(),
                          activation=None)
    relu3 = tf.nn.leaky_relu(fc3, alpha=alpha)

    # bn3 = tf.layers.batch_normalization(relu3, axis=-1, center=True, scale=True,
    #                                     training=(mode == tf.estimator.ModeKeys.TRAIN))
    dropout3 = tf.layers.dropout(
        inputs=relu3, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout3, units=num_unique_classes)
    # Network construction done!

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
        batch_size=1,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    return eval_results
