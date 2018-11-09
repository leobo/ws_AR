import os
import random

import numpy as np
import tensorflow as tf
from sklearn import preprocessing

start_learning_rate = 0.0005

tra_data_splits = 15
eva_data_splits = 8

# 8 - 64
batch_size = 128
alpha = 0.2
num_unique_classes = 101
test_data_ratio = 0.3
epech_decay = int(21847 * tra_data_splits / batch_size) * 5
decay_rate = 0.9
total_epoch = 600
train_epoch = 5
dropout_rate = 0.9
channel = 3
layer_num = 2
clip_len = 16
dim = 2

temp_train = "/home/boy2/UCF101/ucf101_dataset/features/temp/train"
temp_test = "/home/boy2/UCF101/ucf101_dataset/features/temp/test"

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == '__main__':
    output_rbg = np.reshape(np.arange(5 * 1024), newshape=(5, 1024))
    a = tf.stack([output_rbg, output_rbg, output_rbg], axis=1)
    ave_output = tf.reduce_mean(tf.stack([output_rbg, output_rbg, output_rbg], axis=1), axis=1)
    # ave_output = tf.reshape(ave_output, shape=[5, 1024])
    t = tf.Session().run(ave_output)
    print()


def convNet(inputs, mode):
    conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             use_bias=True, kernel_size=[clip_len / 2, 1], padding="valid", activation=tf.nn.relu)

    conv_bn1 = tf.layers.batch_normalization(conv1, training=(mode == tf.estimator.ModeKeys.TRAIN))

    conv2 = tf.layers.conv2d(inputs=conv_bn1, filters=16, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             use_bias=True, kernel_size=[clip_len / 2 + 1, 1], padding="valid", activation=tf.nn.relu)

    conv_bn2 = tf.layers.batch_normalization(conv2, training=(mode == tf.estimator.ModeKeys.TRAIN))

    conv3 = tf.layers.conv2d(inputs=conv_bn2, filters=8, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             use_bias=True, kernel_size=[1, 1], padding="valid", activation=tf.nn.relu)

    conv_bn3 = tf.layers.batch_normalization(conv3, training=(mode == tf.estimator.ModeKeys.TRAIN))

    flat = tf.reshape(conv_bn3, [-1, 1 * 2048 * 8])

    fc1 = tf.layers.dense(inputs=flat, units=2048, activation=tf.nn.relu)
    fc_bn1 = tf.layers.batch_normalization(inputs=fc1, training=(mode == tf.estimator.ModeKeys.TRAIN))
    dropoutfc1 = tf.layers.dropout(
        inputs=fc_bn1, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN))

    fc2 = tf.layers.dense(inputs=dropoutfc1, units=1024, activation=tf.nn.relu)
    fc_bn2 = tf.layers.batch_normalization(inputs=fc2, training=(mode == tf.estimator.ModeKeys.TRAIN))
    dropoutfc2 = tf.layers.dropout(
        inputs=fc_bn2, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Logits Layer
    logits = tf.layers.dense(inputs=dropoutfc2, units=num_unique_classes)

    return logits


def nn_classifier(features, labels, mode, params):
    input_layer = tf.reshape(features['rgb'], [-1, clip_len, 2048, 3])
    input_layer = tf.cast(input_layer, tf.float32, name='input_layer')
    input_rgb, input_u, input_v = tf.split(input_layer, 3, axis=-1)

    output_rgb = convNet(input_rgb, mode)
    with tf.device('/GPU:1'):
        output_u = convNet(input_u, mode)
        output_v = convNet(input_v, mode)

    logits = tf.reduce_mean(tf.stack([output_rgb, output_u, output_v], axis=1), axis=1)

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
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[summary_hook])

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


def load_data(names, path):
    if type(names) is not str:
        return [np.load(os.path.join(path, name + '.npy')) for name in names]
    else:
        return np.load(os.path.join(path, names + '.npy'))


def split_data(data_list, num_splits):
    data_list['label'] = preprocessing.LabelEncoder().fit_transform(data_list['label'])
    data_list['label'] = np.array([ele + 1 for ele in data_list['label']])

    # split data into num_splits parts
    data_splits = np.array_split(np.array(data_list['data']), num_splits)
    label_splits = np.array_split(np.array(data_list['label']), num_splits)
    return data_splits, label_splits


def create_video_clips(data, label=None, rgb_len=None, clip_len=16):
    data_clips = []
    if label is not None and rgb_len is None:
        label_clips = []
        rgb_len = []
        for d, l in zip(data, label):
            clip = [d[i: i + clip_len] if i + clip_len < len(d) else d[len(d) - clip_len: len(d)] for i
                    in range(0, len(d) - int(clip_len / 2), int(clip_len / 2))]
            data_clips += clip
            label_clips += [l for i in range(len(clip))]
            rgb_len.append(len(clip))
        return np.array(data_clips), np.array(label_clips), np.array(rgb_len)
    else:
        for d, l in zip(data, rgb_len):
            clip = [d[i: i + clip_len] if i + clip_len < len(d) else d[len(d) - clip_len: len(d)] for i
                    in range(0, len(d) - int(clip_len / 2), int(clip_len / 2))]
            if len(clip) < l:
                clip.append(clip[-1])
            data_clips += clip
        return np.array(data_clips)


def min_len(train_list, test_list, image_feature_path):
    m = 1000
    train_rgb = load_data(train_list, image_feature_path)
    for i in train_rgb:
        if m > len(i):
            m = len(i)

    test_rgb = load_data(test_list, image_feature_path)
    for i in test_rgb:
        if m > len(i):
            m = len(i)
    print(m)


def split_and_store(name_list, image_feature_path, flow_feature_path_u, flow_feature_path_v,
                    store_path):
    i = 0
    full_path_list = []
    full_label_list = []
    for name, label in zip(name_list['data'], name_list['label']):
        rgb = load_data(name, image_feature_path)
        u = load_data(name, flow_feature_path_u)
        v = load_data(name, flow_feature_path_v)
        rgb, rgb_label_list, rgb_len = create_video_clips([rgb], [label], clip_len=clip_len)
        u = create_video_clips([u], rgb_len=rgb_len, clip_len=clip_len)
        v = create_video_clips([v], rgb_len=rgb_len, clip_len=clip_len)
        train_data = np.stack((rgb, u, v), axis=-1)
        for d, l in zip(train_data, rgb_label_list):
            path = os.path.join(store_path, str(i))
            np.save(path, d)
            full_path_list.append(path)
            full_label_list.append(l)
            i += 1
    return {'data': full_path_list, 'label': full_label_list}


def classify(train_list, test_list, image_feature_path, flow_feature_path_u, flow_feature_path_v):
    # min_len(train_list['data'], test_list['data'], image_feature_path)

    # create and store video clips
    train_list = split_and_store(train_list, image_feature_path, flow_feature_path_u, flow_feature_path_v,
                                 temp_train)
    test_list = split_and_store(test_list, image_feature_path, flow_feature_path_u, flow_feature_path_v,
                                temp_test)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    config.allow_soft_placement = True
    with tf.device('/GPU:0'):
        sess = tf.Session(config=config)

        tensors_to_log = {"accuracy": "acc"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)
        classifier = tf.estimator.Estimator(model_fn=nn_classifier, params=None)
        best_result = 0
        # training
        for i in range(total_epoch):
            # prepare the input data
            print('_____________The epoch ', i, 'start_______________')
            train_list['data'], train_list['label'] = shuffle(train_list['data'], train_list['label'])
            train_data_splits, train_label_splits = split_data(train_list, tra_data_splits)
            j = 0
            for d_list, l_list in zip(train_data_splits, train_label_splits):
                print('_______The epoch ', i, 'split', j, 'start_______________')
                # weighted sum
                # train_rgb = calWeightedSum.calculate_weightedsum_fixed_len(train_rgb, dim, clip_len)
                # train_u = calWeightedSum.calculate_weightedsum_fixed_len(train_u, dim, clip_len)
                # train_v = calWeightedSum.calculate_weightedsum_fixed_len(train_v, dim, clip_len)
                train_data = load_data(d_list, temp_train)

                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                train_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"rgb": np.array(train_data)},
                    y=l_list,
                    batch_size=batch_size,
                    num_epochs=1,
                    shuffle=True,
                    num_threads=32)

                classifier.train(
                    input_fn=train_input_fn,
                    steps=None,
                    hooks=[logging_hook]
                )
                print('_______The epoch ', i, 'split', j, 'finished_______________')
                j += 1
            print('_____________The epoch ', i, 'finished_______________')

            if i % train_epoch == 0:
                # evaluation
                print("______EVALUATION________")
                eval_acc = 0
                test_data_splits, test_label_splits = split_data(test_list, eva_data_splits)
                for d_list, l_list in zip(test_data_splits, test_label_splits):
                    # weighted sum
                    # test_rgb = calWeightedSum.calculate_weightedsum_fixed_len(test_rgb, dim, clip_len)
                    # test_u = calWeightedSum.calculate_weightedsum_fixed_len(test_u, dim, clip_len)
                    # test_v = calWeightedSum.calculate_weightedsum_fixed_len(test_v, dim, clip_len)
                    test_data = load_data(d_list, temp_test)

                    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                        x={"rgb": np.array(test_data)},
                        y=l_list,
                        batch_size=batch_size,
                        num_epochs=1,
                        shuffle=False
                    )
                    eval = classifier.evaluate(input_fn=eval_input_fn)
                    eval_acc += eval['accuracy']
                eval_acc /= eva_data_splits
                print("Accuracy for evaluation is:", eval_acc)
                if eval_acc < best_result * 0.8:
                    print(best_result)
                    return best_result
                elif eval_acc > best_result:
                    best_result = eval_acc
                print("______EVALUATION DONE________")
