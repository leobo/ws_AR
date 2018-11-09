import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
#
# start_learning_rate = 0.0005
# # 8 - 64
# batch_size = 512
# dropout_rate = 0.9
# alpha = 0.2
# num_unique_classes = 101
# test_data_ratio = 0.3
# epech_decay = 120
# decay_rate = 0.1
# total_epoch = 600
# train_epoch = 5

start_learning_rate = 0.0005
# 8 - 64
batch_size = 128
alpha = 0.2
num_unique_classes = 101
test_data_ratio = 0.3
epech_decay = int(9536 / batch_size) * 5
decay_rate = 0.9
total_epoch = 600
train_epoch = 5
dropout_rate = 0.9
channel = 2
layer_num = 2
clip_len = 16
dim = 2
tra_data_splits = 1
eva_data_splits = 5


def nn_classifier(features, labels, mode, params):
    input_layer = tf.reshape(features['x'],
                             [-1, params['feature_shape'][0], params['feature_shape'][1], params['feature_shape'][2]])
    input_layer = tf.cast(input_layer, tf.float32, name='input_layer')

    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             use_bias=True, kernel_size=[1, 1], padding="valid", activation=tf.nn.selu, name='conv1')

    conv_bn1 = tf.layers.batch_normalization(conv1, training=(mode == tf.estimator.ModeKeys.TRAIN), name='conv_bn1')

    conv2 = tf.layers.conv2d(inputs=conv_bn1, filters=16, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             use_bias=True, kernel_size=[2, 1], padding="valid", activation=tf.nn.selu, name='conv2')

    conv_bn2 = tf.layers.batch_normalization(conv2, training=(mode == tf.estimator.ModeKeys.TRAIN), name='conv_bn2')

    conv3 = tf.layers.conv2d(inputs=conv_bn2, filters=8, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             use_bias=True, kernel_size=[1, 1], padding="valid", activation=tf.nn.selu, name='conv3')

    conv_bn3 = tf.layers.batch_normalization(conv3, training=(mode == tf.estimator.ModeKeys.TRAIN), name='conv_bn3')

    flat = tf.reshape(conv_bn3, [-1, 1 * params['feature_shape'][1] * 8], name='flat')

    # fc1 = tf.layers.dense(inputs=flat, units=2048, activation=tf.nn.selu, name='fc1')
    # fc_bn1 = tf.layers.batch_normalization(inputs=fc1, training=(mode == tf.estimator.ModeKeys.TRAIN), name='fc_bn1')
    # dropoutfc1 = tf.layers.dropout(
    #     inputs=fc_bn1, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN), name='dropoutfc1')

    fc2 = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.selu, name='fc2')
    fc_bn2 = tf.layers.batch_normalization(inputs=fc2, training=(mode == tf.estimator.ModeKeys.TRAIN), name='fc_bn2')
    dropoutfc2 = tf.layers.dropout(
        inputs=fc_bn2, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN), name='dropoutfc2')

    # Logits Layer
    logits = tf.layers.dense(inputs=dropoutfc2, units=num_unique_classes, name='logits')

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

    # decayed_learning_rate = learning_rate *
    #     decay_rate ^ (global_step / decay_steps)
    # learning_rate = tf.train.exponential_decay(
    #     start_learning_rate,  # Base learning rate.
    #     tf.train.get_global_step() * batch_size,  # Current index into the dataset.
    #     params['train_set_size'] * epech_decay,  # Decay step.
    #     decay_rate,  # Decay rate.
    #     staircase=True)

    learning_rate = tf.train.exponential_decay(
        learning_rate=start_learning_rate,  # Base learning rate.
        global_step=tf.train.get_global_step(),  # Current index into the dataset.
        decay_steps=epech_decay,  # Decay step.
        decay_rate=decay_rate,  # Decay rate.
        staircase=True)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            tf.summary.scalar('loss', loss)

            summary_hook = tf.train.SummarySaverHook(
                save_steps=100, output_dir='/tmp/tf', summary_op=tf.summary.merge_all()
            )
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[summary_hook])

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def classify(train_data, train_labels, eval_data, eval_labels):
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1000000)

    params = {"train_set_size": len(train_data), "feature_shape": train_data[0].shape}
    mnist_classifier = tf.estimator.Estimator(model_fn=nn_classifier, params=params)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=batch_size,
        num_epochs=train_epoch,
        shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        batch_size=batch_size,
        num_epochs=1,
        shuffle=False)

    best_result = 0
    while True:
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=None,
            hooks=[logging_hook])

        print("_________EVALUATION START___________")
        eval = mnist_classifier.evaluate(input_fn=eval_input_fn)
        if eval['accuracy'] < best_result * 0.8:
            print(best_result)
            return best_result
        elif eval['accuracy'] > best_result:
            best_result = eval['accuracy']
        print("_________EVALUATION DONE___________")
