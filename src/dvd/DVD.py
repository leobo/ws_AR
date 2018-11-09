import tensorflow as tf

# tf.logging.set_verbosity(tf.logging.INFO)

start_learning_rate = 0.0005
# 8 - 64
batch_size = 256
dropout_rate = 0.9
alpha = 0.2
num_unique_classes = 101
test_data_ratio = 0.3
epech_decay = 20
decay_rate = 0.9
total_epoch = 600
train_epoch = 10


def rnn_time_model():
    with tf.name_scope('inputs'):
        input_layer = tf.placeholder(dtype=tf.float32, shape=[None, 2048, None], name='input_layer')
        labels = tf.placeholder(dtype=tf.int32, shape=[None], name='labels')
        # mode = tf.placeholder(dtype=tf.int32)
        batch_num = tf.placeholder(dtype=tf.int32, name='batch_num')
        train_set_size = tf.placeholder(dtype=tf.int32, name='train_set_size')

    with tf.name_scope('Conv'):
        with tf.device('/GPU:1'):
            conv_1 = tf.layers.conv1d(input_layer, filters=32, kernel_size=1, strides=1, padding='valid')
            logits = tf.layers.dense(inputs=conv_1, units=num_unique_classes, name='logits')

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=num_unique_classes)
    with tf.name_scope('loss'):
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1, output_type=tf.int32),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    learning_rate = tf.train.exponential_decay(
        start_learning_rate,  # Base learning rate.
        batch_num * batch_size,  # Current index into the dataset.
        train_set_size * epech_decay,  # Decay step.
        decay_rate,  # Decay rate.
        staircase=True)

    # Configure the Training Op (for TRAIN mode)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    tf.summary.scalar('loss', loss)
    equality = tf.equal(predictions["classes"], labels)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32), name='accuracy')
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()
    return train_op, accuracy, loss, merged


def classify(train_data, train_labels, eval_data, eval_labels):
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100000)

    params = {"train_set_size": len(train_data), "feature_len": len(train_data[0])}
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
    for i in range(int(total_epoch/train_epoch)):
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=None,
            hooks=[logging_hook])

        eval = mnist_classifier.evaluate(input_fn=eval_input_fn)
        if eval['accuracy'] < best_result*0.8:
            print(best_result)
            return best_result
        elif eval['accuracy'] > best_result:
            best_result = eval['accuracy']

if __name__ == '__main__':
    b = tf.concat([tf.constant([1,2]), tf.constant([3,4])], axis=0)
    a = tf.Session().run(b)
    print(a)