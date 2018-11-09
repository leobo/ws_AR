import tensorflow as tf


class nn_classifier(object):

    def __init__(self):
        """"""

    # The convolution layer
    def conv_layer(self, input, channel_in, channel_out, name="conv"):
        """
        Define the convolution layer
        :param input:
        :param channel_in:
        :param channel_out:
        :param name:
        :return:
        """
        with tf.name_scope(name):
            # The weights with random values
            w = tf.Variable(tf.truncated_normal([5, 5, channel_in, channel_out], stddev=0.1), name="W")
            # The bias with initial value 0.1
            b = tf.Variable(tf.constant(0.1, shape=[channel_out]), name="B")
            # About the strides: 1 step in batch dimension, 1 step in width, 1 step in height and 1 step in color channel
            conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
            # The activation function
            act = tf.nn.relu(conv + b)
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
            # The convolution layer followed by a max pooling layer
            return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # The fully connection layer
    def fc_layer(self, input, size_in, size_out, name="fc"):
        """
        Define the fully connected layer
        :param input:
        :param size_in:
        :param size_out:
        :param name:
        :return:
        """
        with tf.name_scope(name):
            # The weights
            w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
            # The bias
            b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
            # The activation function
            act = tf.nn.relu(tf.matmul(input, w) + b)
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
            return act

    def build(self, x_size, num_unique_classes, start_learning_rate, batch_size, total_number_data, alpha, dropout_rate,
              test_ratio, epech_decay, decay_rate):
        """
        Build the model
        :return:
        """
        x = tf.placeholder(tf.float32, shape=[None, x_size], name="x")
        y = tf.placeholder(tf.float32, shape=[None, ], name="y")
        mode = tf.placeholder(tf.bool, name='mode')

        fc0 = tf.layers.dense(inputs=x, units=3000, activation=None)
        bn0 = tf.layers.batch_normalization(fc0, axis=1, center=True, scale=True,
                                            training=mode == tf.estimator.ModeKeys.TRAIN)
        relu0 = tf.nn.leaky_relu(fc0, alpha=alpha)

        fc1 = tf.layers.dense(inputs=relu0, units=1250, activation=None)
        bn1 = tf.layers.batch_normalization(fc1, axis=1, center=True, scale=True,
                                            training=mode == tf.estimator.ModeKeys.TRAIN)
        relu1 = tf.nn.leaky_relu(fc1, alpha=alpha)

        fc2 = tf.layers.dense(inputs=relu1, units=625, activation=None)
        bn2 = tf.layers.batch_normalization(fc2, axis=1, center=True, scale=True,
                                            training=mode == tf.estimator.ModeKeys.TRAIN)
        relu2 = tf.nn.leaky_relu(fc2, alpha=alpha)
        dropout2 = tf.layers.dropout(
            inputs=relu2, rate=dropout_rate, training=mode)

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout2, units=num_unique_classes)

        # Define the loss function
        with tf.name_scope("cross_entropy"):
            onehot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=num_unique_classes)
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels, logits=logits)
            tf.summary.scalar("cross_entropy", loss)

        # Calculate the accuracy
        with tf.name_scope("accuracy"):
            # accuracy = tf.metrics.accuracy(
            #     labels=y, predictions=tf.argmax(input=logits, axis=1, name='pred'), name='acc')

            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(onehot_labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Define the optimization algorithm
        learning_rate = tf.train.exponential_decay(
            start_learning_rate,  # Base learning rate.
            global_step * batch_size,  # Current index into the dataset.
            total_number_data * (1 - test_ratio) * epech_decay,  # Decay step.
            decay_rate,  # Decay rate.
            staircase=True)

        with tf.name_scope("train"):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # Merge all summary together
        summ = tf.summary.merge_all()

        writer = tf.summary.FileWriter("/home/boy2/UCF101/checkpoint")
        # writer.add_graph(sess.graph)

        return train_step, summ, writer, accuracy, loss, learning_rate, global_step

    def train(self):
        # tf.reset_default_graph()
        clr = nn_classifier()
        train_step, summ, writer, accuracy, loss, learning_rate, global_step = clr.build(len(des[0]), num_unique_classes,
                                                                                         start_learning_rate=start_learning_rate,
                                                                                         batch_size=batch_size,
                                                                                         total_number_data=len(
                                                                                             train_labels), alpha=alpha,
                                                                                         dropout_rate=dropout_rate,
                                                                                         test_ratio=test_data_ratio,
                                                                                         epech_decay=epech_decay,
                                                                                         decay_rate=decay_rate)
        sess = tf.Session()
        # init_op = tf.initialize_all_variables()
        # sess.run(init_op)
        writer.add_graph(sess.graph)

        # Save the model status during training
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        batch = create_batches(train_data, train_labels, batch_size)
        temp_epech = epech
        while temp_epech != 0:
            for i in range(int(np.ceil(len(train_data) / batch_size))):
                idx = i % len(batch)
                # Write the log
                [train_accuracy, train_loss, s, lr, gs, _] = sess.run(
                    [accuracy, loss, summ, learning_rate, global_step, train_step],
                    feed_dict={'x:0': batch[0][idx], 'y:0': batch[1][idx], 'mode:0': True})
                if train_loss < min_loss:
                    min_loss = train_loss
                    saver.save(sess, os.path.join("/home/boy2/UCF101/checkpoint", "model.ckpt"), gs)
                writer.add_summary(s, gs)
                print("Accuracy in step ", gs, "is: ", train_accuracy, "loss is: ", train_loss, "learning rate: ", lr)
            temp_epech -= 1

        # saver.restore(sess, tf.train.latest_checkpoint("/home/boy2/UCF101/checkpoint"))

        # batch = create_batches(eval_data, eval_labels, batch_size)
        [test_accuracy, test_loss] = sess.run([accuracy, loss],
                                              feed_dict={'x:0': eval_data, 'y:0': eval_labels, 'mode:0': False})
        print("Testing: Accuracy is: ", test_accuracy, "loss is ", test_loss)

        #############################################
        des, tar = read_surfWS_tar(WS_path)
        des, tar = shuffle(des, tar)
        des = preprocessing.normalize(np.array(des), norm='l2')
        tar = preprocessing.LabelEncoder().fit_transform(np.array(tar))
        num_unique_classes = len(np.unique(tar))

        print('classifiering')

        train_data, eval_data, train_labels, eval_labels = model_selection.train_test_split(des, tar,
                                                                                            test_size=test_data_ratio,
                                                                                            random_state=42)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
