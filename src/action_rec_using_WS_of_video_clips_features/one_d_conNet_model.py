# max %86 with video clip of cropped video

import random

import numpy as np
import tensorflow as tf
from sklearn import preprocessing

start_learning_rate = 0.001
# 8 - 64
batch_size = 256
alpha = 0.2
num_unique_classes = 101
test_data_ratio = 0.3

decay_rate = 0.5
total_epoch = 650
train_epoch = 30
epech_decay = int(np.ceil(269894 / batch_size)) * train_epoch
dropout_rate = 0.9
channel = 3
layer_num = 2
clip_len = 16
dim = 2
tra_data_splits = 10
eva_data_splits = 5

tf.logging.set_verbosity(tf.logging.INFO)


def nn_classifier_estimator(features, labels, mode, params):
    with tf.device('/device:GPU:1'):
        input_layer = tf.reshape(features['rgb'],
                                 [-1, 3, 2048*2, 1])
        input_layer = tf.cast(input_layer, tf.float32, name='input_layer')

        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 use_bias=True, kernel_size=[1, 1], padding="valid", activation=None, name='conv1')
        conv_bn1 = tf.layers.batch_normalization(conv1, training=(mode == tf.estimator.ModeKeys.TRAIN), name='conv_bn1')
        conv_act1 = tf.nn.selu(conv_bn1, name="conv_act1")

        conv2 = tf.layers.conv2d(inputs=conv_act1, filters=16,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 use_bias=True, kernel_size=[3, 1], padding="same", activation=None, name='conv2')
        conv_bn2 = tf.layers.batch_normalization(conv2, training=(mode == tf.estimator.ModeKeys.TRAIN), name='conv_bn2')
        conv_act2 = tf.nn.selu(conv_bn2, name="conv_act2")

        conv3 = tf.layers.conv2d(inputs=conv_act2, filters=8, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 use_bias=True, kernel_size=[6, 1], padding="valid", activation=None, name='conv3')
        conv_bn3 = tf.layers.batch_normalization(conv3, training=(mode == tf.estimator.ModeKeys.TRAIN), name='conv_bn3')
        conv_act3 = tf.nn.selu(conv_bn3, name="conv_act3")

        flat = tf.reshape(conv_act3, [-1, 1*2048*8], name='flat')

        fc1 = tf.layers.dense(inputs=flat, units=1024, activation=None, name='fc1')
        fc_bn1 = tf.layers.batch_normalization(inputs=fc1, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                               name='fc_bn1')
        fc_act1 = tf.nn.selu(fc_bn1, name="fc_act1")
        dropoutfc1 = tf.layers.dropout(
            inputs=fc_act1, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN), name='dropoutfc1')

        # fc2 = tf.layers.dense(inputs=dropoutfc1, units=4096, activation=None, name='fc2')
        # fc_bn2 = tf.layers.batch_normalization(inputs=fc2, training=(mode == tf.estimator.ModeKeys.TRAIN),
        #                                        name='fc_bn2')
        # fc_act2 = tf.nn.selu(fc_bn2, name="fc_act2")
        # dropoutfc2 = tf.layers.dropout(
        #     inputs=fc_act2, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN), name='dropoutfc2')
        #
        # fc3 = tf.layers.dense(inputs=dropoutfc2, units=1024, activation=None, name='fc3')
        # fc_bn3 = tf.layers.batch_normalization(inputs=fc3, training=(mode == tf.estimator.ModeKeys.TRAIN),
        #                                        name='fc_bn3')
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
            "classes": tf.argmax(input=logits, axis=1, output_type=tf.int32),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "accuracy": tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(input=logits, axis=1, output_type=tf.int64), labels), tf.float32),
                name="acc")
        }

        # decayed_learning_rate = learning_rate *
        #     decay_rate ^ (global_step / decay_steps)
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
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                                  training_hooks=[summary_hook])

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])
        }
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def shuffle(data, label):
    temp = list(zip(data, label))
    random.shuffle(temp)
    data, label = zip(*temp)
    return list(data), list(label)


def _parse_fn(example):
    """Parse TFExample records and perform simple data augmentation."""
    example_fmt = {
        "train/feature": tf.FixedLenFeature([], tf.string),
        "train/label": tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(example, example_fmt)
    feature = tf.decode_raw(parsed["train/feature"], tf.float64)
    feature = tf.reshape(feature, [3, 2048, 2])
    # label = tf.reshape(parsed["train/label"], 1)
    return feature, parsed['train/label']


def _input_fn(filenames):
    dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes=126000)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(map_func=_parse_fn)
    dataset = dataset.batch(batch_size=batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return {'rgb': features}, labels


def classify(train_list, test_list):
    # min_len(train_list['data'], test_list['data'], image_feature_path)

    # prepare the input data
    # train_list['data'], train_list['label'] = shuffle(train_list['data'], train_list['label'])
    # train_data_splits, train_label_splits = split_data(train_list, tra_data_splits)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    config.allow_soft_placement = True

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    tensors_to_log = {"accuracy": "acc"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)
    classifier = tf.estimator.Estimator(model_fn=nn_classifier_estimator, params=None)
    best_result = 0

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"rgb": train_list['data']},
        y=train_list['label'],
        batch_size=batch_size,
        num_epochs=1,
        shuffle=True,
        num_threads=32
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"rgb": test_list['data']},
        y=test_list['label'],
        batch_size=batch_size,
        num_epochs=1,
        shuffle=False,
        num_threads=32
    )

    # training
    for _ in range(total_epoch):
        classifier.train(
            input_fn=train_input_fn,
            steps=int(141994 * train_epoch / batch_size),
            hooks=[logging_hook]
        )

        # evaluation
        print("______EVALUATION________")
        eval = classifier.evaluate(input_fn=eval_input_fn)
        eval_acc = eval['accuracy']
        print("Accuracy for evaluation is:", eval_acc)
        if eval_acc < best_result * 0.8:
            print(best_result)
            return best_result
        elif eval_acc > best_result:
            best_result = eval_acc
        print("______EVALUATION DONE________")


def load_data(paths):
    data = []
    labels = []
    for p in paths:
        contents = np.load(p + '.npy')
        contents = contents.item()
        for key, value in contents.items():
            data.append(value)
            labels.append(key)
    return {'data': np.array(data), 'label': labels}


def split_data(data_list, num_splits):
    data_list['label'] = preprocessing.LabelEncoder().fit_transform(data_list['label'])
    data_list['label'] = np.array([ele + 1 for ele in data_list['label']])

    # split data into num_splits parts
    data_splits = np.array_split(np.array(data_list['data']), num_splits)
    label_splits = np.array_split(np.array(data_list['label']), num_splits)
    return data_splits, label_splits


def parse_fn(example):
    "Parse TFExample records and perform simple data augmentation."
    example_fmt = {
        "train/feature": tf.FixedLenFeature([], tf.string),
        "train/label": tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(example, example_fmt)
    feature = tf.decode_raw(parsed["train/feature"], tf.float64)
    feature = tf.reshape(feature, [3, 2048*2, 1])
    # label = tf.reshape(parsed["train/label"], 1)
    return feature, parsed['train/label']


def input_fn_train(filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(map_func=parse_fn)
    dataset = dataset.batch(batch_size=batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return {'rgb': features}, labels


def input_fn_eval(filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(map_func=parse_fn)
    dataset = dataset.batch(batch_size=batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return {'rgb': features}, labels


def classify_tfrecord(train_records, test_records):
    with tf.device('/device:GPU:1'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        tensors_to_log = {"accuracy": "acc"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)
        classifier = tf.estimator.Estimator(model_fn=nn_classifier_estimator, params=None)
        best_result = 0

        # for _ in range(53):
        #     data, label = input_fn_eval(test_records)
        #     d = sess.run(data)

        # training and evaluating input functions
        train_input_fn = lambda: input_fn_train([train_records])
        eval_input_fn = lambda: input_fn_eval([test_records])

        # training
        # 269894 -> crop video with length 25 and overlap 0.75
        for _ in range(1, total_epoch):
            classifier.train(
                input_fn=train_input_fn,
                steps=int(np.ceil(269894 / batch_size)) * train_epoch,
                hooks=[logging_hook]
            )

            # evaluation
            print("______EVALUATION________")
            eval = classifier.evaluate(input_fn=eval_input_fn, steps=int(np.ceil(105163 / batch_size)))
            eval_acc = eval['accuracy']
            print("Accuracy for evaluation after training epoch", _*train_epoch, "is:", eval_acc)
            if eval_acc < best_result * 0.8:
                print("__________The best evaluation result is___________", best_result)
                # return best_result
            elif eval_acc > best_result:
                best_result = eval_acc
            with open("/home/boy2/UCF101/ucf101_dataset/exp_results/res_may_27", "a") as text_file:
                text_file.writelines("Evaluation accuracy after training epoch %s is: %s \n" % (_*train_epoch, eval_acc))
            print("______EVALUATION DONE________")
