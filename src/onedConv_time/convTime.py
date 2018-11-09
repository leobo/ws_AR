import os
import random

import numpy as np
import tensorflow as tf
from sklearn import preprocessing

from RnnCell.conv_Rnn import ConvRnnCell

start_learning_rate = 0.001
# 8 - 64
batch_size = 32
alpha = 0.2
num_unique_classes = 101
test_data_ratio = 0.3
epech_decay = int(9536 / batch_size) * 5
decay_rate = 0.9
total_epoch = 600
train_epoch = 5
dropout = 0.9
channel = 3
layer_num = 2
clip_len = 25
eva_data_splits = 8
tra_data_splits = 15


# tf.logging.set_verbosity(tf.logging.INFO)

def make_cell(num_units, activation, dropout_rate, mode):
    cell = ConvRnnCell(num_units=num_units, feature_len=2048, channel=channel, mode=mode)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1 - dropout_rate, seed=42)
    return cell


def rnn_time_model(features, labels, mode, params):
    mask_idx_fw = tf.stack(
        [tf.range(tf.shape(features['rgb'])[0]),
         tf.fill(dims=[tf.shape(features['rgb'])[0]], value=clip_len - 1)], axis=1)
    mask_idx_bw = tf.stack(
        [tf.range(tf.shape(features['rgb'])[0]), tf.zeros(shape=[tf.shape(features['rgb'])[0]], dtype=tf.int32)],
        axis=1)

    with tf.device('/GPU:0'):
        if mode == tf.estimator.ModeKeys.TRAIN:
            d = tf.constant(0.5, dtype=tf.float32)
        else:
            d = tf.constant(0, dtype=tf.float32)
        input_layer = tf.reshape(features['rgb'],
                                 shape=[tf.shape(features['rgb'])[0], tf.shape(features['rgb'])[1], 2048 * channel])
        cell_fw = tf.nn.rnn_cell.MultiRNNCell(cells=[make_cell(2048, tf.nn.selu, d, mode) for _ in range(layer_num)],
                                              state_is_tuple=True)
        cell_bw = tf.nn.rnn_cell.MultiRNNCell(cells=[make_cell(2048, tf.nn.selu, d, mode) for _ in range(layer_num)],
                                              state_is_tuple=True)
        initial_state_fw = cell_fw.zero_state(tf.shape(features['rgb'])[0], tf.float32)
        initial_state_bw = cell_bw.zero_state(tf.shape(features['rgb'])[0], tf.float32)

    with tf.device('/GPU:1'):
        out, state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                     cell_bw,
                                                     input_layer,
                                                     sequence_length=None,
                                                     initial_state_fw=initial_state_fw,
                                                     initial_state_bw=initial_state_bw,
                                                     dtype=tf.float32, scope='rnn_networks')
        # out, state = tf.nn.dynamic_rnn(cell_fw, input_layer, initial_state=initial_state_fw, dtype=tf.float32,
        #                                scope='rnn_networks')
    with tf.device('/GPU:0'):
        out_fw = tf.gather_nd(out[0], mask_idx_fw)
        out_bw = tf.gather_nd(out[1], mask_idx_bw)
        bn_fw = tf.layers.batch_normalization(out_fw, training=(mode == tf.estimator.ModeKeys.TRAIN), name='bn_fw')
        bn_bw = tf.layers.batch_normalization(out_bw, training=(mode == tf.estimator.ModeKeys.TRAIN), name='bn_bw')
        com = tf.nn.selu(tf.multiply(bn_fw, bn_bw))
        fc = tf.layers.dense(inputs=com, units=512, activation=tf.nn.selu, name='fc')
        fc_bn = tf.layers.batch_normalization(inputs=fc, training=(mode == tf.estimator.ModeKeys.TRAIN), name='fc_bn')
        dropoutfc = tf.layers.dropout(
            inputs=fc_bn, rate=dropout, training=(mode == tf.estimator.ModeKeys.TRAIN), name='dropoutfc')
        logits = tf.layers.dense(inputs=dropoutfc, units=num_unique_classes, name='logits')

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

        if mode == tf.estimator.ModeKeys.TRAIN:
            # Configure the Training Op (for TRAIN mode)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            # tf.summary.scalar('loss', loss)

            # summary_hook = tf.train.SummarySaverHook(
            #     save_steps=100, output_dir='/tmp/tf', summary_op=tf.summary.merge_all()
            # )
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, predictions=a)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def shuffle(data, label):
    temp = list(zip(data, label))
    random.shuffle(temp)
    data, label = zip(*temp)
    return list(data), list(label)


def padding(data_batch, max_len=None):
    # max_len = 846
    # max_len = 1776
    paded_data = []
    seq_len = []
    if max_len == None:
        max_len = 0
        for sample in data_batch:
            seq_len.append(len(sample))
            if len(sample) > max_len:
                max_len = len(sample)
    for sample in data_batch:
        paded_data.append(np.swapaxes(np.swapaxes(np.pad(sample, [[0, max_len - len(sample)], [0, 0]], 'constant',
                                                         constant_values=(0, 0)), 0, 1), 0, 1))
    return paded_data, seq_len


def load_data(names, path):
    return [np.load(os.path.join(path, name + '.npy')) for name in names]


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


def data_batch_gen(data, label):
    data_queue, label_queue = tf.train.slice_input_producer(
        [tf.convert_to_tensor(data, dtype=tf.float32),
         tf.convert_to_tensor(label, dtype=tf.int32)],
        capacity=batch_size,
        num_epochs=train_epoch,
        shuffle=True,
    )

    data_batch, label_batch = tf.train.batch(
        [data_queue, label_queue],
        batch_size=batch_size,
        dynamic_pad=False,
        allow_smaller_final_batch=True,
        enqueue_many=False,
        num_threads=1
    )
    return data_batch, label_batch


def classify(train_list, test_list, image_feature_path, flow_feature_path_u, flow_feature_path_v):
    # prepare the input data
    train_list['data'], train_list['label'] = shuffle(train_list['data'], train_list['label'])
    train_data_splits, train_label_splits = split_data(train_list, tra_data_splits)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    config.allow_soft_placement = True
    with tf.device('/GPU:0'):
        sess = tf.Session(config=config)

        tensors_to_log = {"accuracy": "acc"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=10000000)
        params = {"train_set_size": len(train_list)}
        classifier = tf.estimator.Estimator(model_fn=rnn_time_model, params=params)
        best_result = 0
        # training
        for i in range(total_epoch):
            for d_list, l_list in zip(train_data_splits, train_label_splits):
                d_list, l_list = shuffle(d_list, l_list)
                train_rgb = load_data(d_list, image_feature_path)
                train_u = load_data(d_list, flow_feature_path_u)
                train_v = load_data(d_list, flow_feature_path_v)
                train_rgb, rgb_label_list, rgb_len = create_video_clips(train_rgb, l_list, clip_len=clip_len)
                train_u = create_video_clips(train_u, rgb_len=rgb_len, clip_len=clip_len)
                train_v = create_video_clips(train_v, rgb_len=rgb_len, clip_len=clip_len)
                train_data = np.stack((train_rgb, train_u, train_v), axis=-1)

                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                train_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"rgb": train_data},
                    y=rgb_label_list,
                    batch_size=batch_size,
                    num_epochs=train_epoch,
                    shuffle=True)

                classifier.train(
                    input_fn=train_input_fn,
                    steps=None,
                    hooks=[logging_hook]
                )

            # evaluation
            print("______EVALUATION________")
            eval_acc = 0
            test_data_splits, test_label_splits = split_data(test_list, eva_data_splits)
            for d_list, l_list in zip(test_data_splits, test_label_splits):
                test_rgb = load_data(d_list, image_feature_path)
                test_u = load_data(d_list, flow_feature_path_u)
                test_v = load_data(d_list, flow_feature_path_v)
                test_rgb, test_labels, rgb_len = create_video_clips(test_rgb, l_list, clip_len=clip_len)
                test_u = create_video_clips(test_u, rgb_len=rgb_len, clip_len=clip_len)
                test_v = create_video_clips(test_v, rgb_len=rgb_len, clip_len=clip_len)
                test_data = np.stack((test_rgb, test_u, test_v), axis=-1)

                eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"rgb": test_data},
                    y=test_labels,
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
