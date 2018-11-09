import datetime

import numpy as np
import tensorflow as tf

# tf.logging.set_verbosity(tf.logging.INFO)
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

start_learning_rate = 0.001
# 8 - 64
batch_size = 512
alpha = 0.2
num_unique_classes = 101
test_data_ratio = 0.3
total_epoch = 650
train_epoch = 15
epech_decay = int(9537 * 2 / batch_size) * train_epoch * 2
decay_rate = 0.9
dropout_rate = 0.9
channel = 3

layer_num = 2
clip_len = 16
dim = 2
tra_data_splits = 1
eva_data_splits = 5


def nn_classifier(features, labels, mode, params):
    input_layer = tf.reshape(features['x'],
                             [-1, params['feature_shape'][0], params['feature_shape'][1], params['feature_shape'][2]]
                             )
    input_layer = tf.cast(input_layer, tf.float32, name='input_layer')
    # input_layer = tf.transpose(input_layer, [0, 1, 3, 2])

    conv1 = tf.layers.conv2d(inputs=input_layer, filters=16, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             use_bias=True, kernel_size=[1, 1], padding="same", activation=None, name='conv1')
    conv_bn1 = tf.layers.batch_normalization(conv1, training=(mode == tf.estimator.ModeKeys.TRAIN), name='conv_bn1')
    conv_act1 = tf.nn.selu(conv_bn1, name="conv_act1")
    # pool1 = tf.layers.average_pooling3d(inputs=conv_act1, pool_size=[2, 2, 2], strides=[1, 2, 2])

    conv2 = tf.layers.conv2d(inputs=conv_act1, filters=16, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             use_bias=True, kernel_size=[3, 1], padding="same", activation=None, name='conv2')
    conv_bn2 = tf.layers.batch_normalization(conv2, training=(mode == tf.estimator.ModeKeys.TRAIN), name='conv_bn2')
    conv_act2 = tf.nn.selu(conv_bn2, name="conv_act2")
    # pool2 = tf.layers.max_pooling2d(inputs=conv_act2, pool_size=[3, 1], strides=[1, 1])

    add_conv1 = tf.layers.conv2d(inputs=input_layer, filters=16,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 use_bias=True, kernel_size=[1, 1], padding="same", activation=None, name='add_conv2')
    add_conv_bn1 = tf.layers.batch_normalization(add_conv1, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                 name='add_conv_bn2')
    add_conv_act1 = tf.nn.selu(add_conv_bn1, name="add_conv_act2")
    # pool2 = tf.layers.max_pooling2d(inputs=conv_act2, pool_size=[3, 1], strides=[1, 1])

    concat = tf.concat([conv_act2, add_conv_act1], 1)

    conv3 = tf.layers.conv2d(inputs=concat, filters=4, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             use_bias=True, kernel_size=[3, 4], padding="same", activation=None, name='conv3')
    conv_bn3 = tf.layers.batch_normalization(conv3, training=(mode == tf.estimator.ModeKeys.TRAIN), name='conv_bn3')
    conv_act3 = tf.nn.selu(conv_bn3, name="conv_act3")
    pool3 = tf.layers.average_pooling2d(inputs=conv_act3, pool_size=[3, 4], strides=[3, 4])

    flat = tf.reshape(pool3, [-1, 2 * 512 * 4], name='flat')

    fc1 = tf.layers.dense(inputs=flat, units=1024, activation=None, name='fc1')
    # fc_bn1 = tf.layers.batch_normalization(inputs=fc1, training=(mode == tf.estimator.ModeKeys.TRAIN), name='fc_bn1')
    # fc_act1 = tf.nn.selu(fc_bn1, name="fc_act1")
    dropoutfc1 = tf.layers.dropout(
        inputs=fc1, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN), name='dropoutfc1')

    # fc2 = tf.layers.dense(inputs=dropoutfc1, units=1024, activation=None, name='fc2')
    # fc_bn2 = tf.layers.batch_normalization(inputs=fc2, training=(mode == tf.estimator.ModeKeys.TRAIN), name='fc_bn2')
    # fc_act2 = tf.nn.selu(fc_bn2, name="fc_act2")
    # dropoutfc2 = tf.layers.dropout(
    #     inputs=fc_act2, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN), name='dropoutfc2')
    #
    # fc3 = tf.layers.dense(inputs=dropoutfc2, units=101, activation=None, name='fc3')
    # fc_bn3 = tf.layers.batch_normalization(inputs=fc3, training=(mode == tf.estimator.ModeKeys.TRAIN), name='fc_bn3')
    # fc_act3 = tf.nn.selu(fc_bn3, name="fc_act3")
    # dropoutfc3 = tf.layers.dropout(
    #     inputs=fc_act3, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN), name='dropoutfc3')

    # Logits Layer
    logits = tf.layers.dense(inputs=dropoutfc1, units=num_unique_classes, name='logits')

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

    # learning_rate = tf.train.exponential_decay(
    #     learning_rate=start_learning_rate,  # Base learning rate.
    #     global_step=tf.train.get_global_step(),  # Current index into the dataset.
    #     decay_steps=epech_decay,  # Decay step.
    #     decay_rate=decay_rate,  # Decay rate.
    #     staircase=True)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=features['lr'][0])
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

    learning_rate = start_learning_rate
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data, "lr": np.array([learning_rate for _ in range(len(train_data))])},
        y=train_labels,
        batch_size=batch_size,
        num_epochs=train_epoch,
        shuffle=True,
        num_threads=1
    )

    best_result = 0
    prev_result = 0
    first84 = True
    first86 = True
    first845 = True
    first85 = True
    first853 = True
    exp_result = "/home/boy2/UCF101/ucf101_dataset/exp_results/res_for_1dconv_classifier_at_" + str(
        datetime.datetime.now())
    for _ in range(1, total_epoch):
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=None,
            hooks=[logging_hook])

        # # evaluation
        # print("______EVALUATION________")
        # eval = mnist_classifier.evaluate(input_fn=eval_input_fn)
        # eval_acc = eval['accuracy']
        # print("Accuracy for evaluation is:", eval_acc)
        # if eval_acc < best_result * 0.5:
        #     print(best_result)
        #     return best_result
        # elif eval_acc > best_result:
        #     best_result = eval_acc
        # print("______EVALUATION DONE________")

        print("_________EVALUATION START___________")
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data, "lr": np.array([learning_rate for _ in range(len(eval_data))])},
            y=eval_labels,
            batch_size=batch_size,
            num_epochs=1,
            shuffle=False,
            num_threads=1
        )
        eval = mnist_classifier.evaluate(input_fn=eval_input_fn)
        with open(exp_result, "a") as text_file:
            text_file.writelines(
                "Evaluation accuracy after training epoch %s is: %s \n" % (_ * train_epoch, eval['accuracy']))
        if eval['accuracy'] >= prev_result:
            if eval['accuracy'] > best_result:
                best_result = eval['accuracy']
        else:
            if eval['accuracy'] < (prev_result * 0.5):
                print("Training will stop, the best result is", best_result)
                # return best_result
            elif (prev_result * 0.5) <= eval['accuracy'] < (prev_result - 0.01) or eval["accuracy"] < \
                    best_result - 0.05:
                learning_rate *= 0.1
                train_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": train_data, "lr": np.array([learning_rate for _ in range(len(train_data))])},
                    y=train_labels,
                    batch_size=batch_size,
                    num_epochs=1,
                    shuffle=True,
                    num_threads=1
                )
                print('The learning rate is decreased to', learning_rate)
            elif eval["accuracy"] > 0.84 and first84 is True:
                learning_rate *= 0.1
                train_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": train_data, "lr": np.array([learning_rate for _ in range(len(train_data))])},
                    y=train_labels,
                    batch_size=batch_size,
                    num_epochs=1,
                    shuffle=True,
                    num_threads=1
                )
                print('The learning rate is decreased to', learning_rate)
                first84 = False
            elif eval["accuracy"] > 0.845 and first845 is True:
                learning_rate *= 0.5
                train_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": train_data, "lr": np.array([learning_rate for _ in range(len(train_data))])},
                    y=train_labels,
                    batch_size=batch_size,
                    num_epochs=1,
                    shuffle=True,
                    num_threads=1
                )
                print('The learning rate is decreased to', learning_rate)
                first845 = False
            elif eval["accuracy"] > 0.85 and first85 is True:
                learning_rate *= 0.1
                train_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": train_data, "lr": np.array([learning_rate for _ in range(len(train_data))])},
                    y=train_labels,
                    batch_size=batch_size,
                    num_epochs=1,
                    shuffle=True,
                    num_threads=1
                )
                print('The learning rate is decreased to', learning_rate)
                first85 = False
            elif eval["accuracy"] > 0.852 and first853 is True:
                learning_rate *= 0.5
                train_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": train_data, "lr": np.array([learning_rate for _ in range(len(train_data))])},
                    y=train_labels,
                    batch_size=batch_size,
                    num_epochs=1,
                    shuffle=True,
                    num_threads=1
                )
                print('The learning rate is decreased to', learning_rate)
                first853 = False
            elif eval["accuracy"] > 0.86 and first86 is True:
                learning_rate *= 0.1
                train_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": train_data, "lr": np.array([learning_rate for _ in range(len(train_data))])},
                    y=train_labels,
                    batch_size=batch_size,
                    num_epochs=1,
                    shuffle=True,
                    num_threads=1
                )
                print('The learning rate is decreased to', learning_rate)
                first86 = False
        prev_result = eval['accuracy']
        print("_________EVALUATION DONE___________")
