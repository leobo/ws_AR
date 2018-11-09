import math
import os
import random
import sys

import numpy as np
import tensorflow as tf
from sklearn import preprocessing

from RnnCell.conv_Rnn import ConvRnnCell

start_learning_rate = 0.001
# 8 - 64
batch_size = 16
alpha = 0.2
num_unique_classes = 101
test_data_ratio = 0.3
epech_decay = 2
decay_rate = 0.9
total_epoch = 600
train_epoch = 5
dropout = 0.5
channel = 3
layer_num = 2
clip_len = 25


def make_cell(num_units, activation, dropout_rate, mode):
    cell = ConvRnnCell(num_units=num_units, feature_len=2048, channel=channel, mode=mode)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1 - dropout_rate, seed=42)
    return cell


def rnn_time_model():
    with tf.name_scope('inputs'):
        input_layer = tf.placeholder(dtype=tf.float32, shape=[None, None, 2048, channel], name='input_layer')
        labels = tf.placeholder(dtype=tf.int32, shape=[None], name='labels')
        mode = tf.placeholder(dtype=tf.string, name='mode')
        epoche = tf.placeholder(dtype=tf.int32, name='epoch')
        seq_len = tf.placeholder(dtype=tf.int32, shape=[None], name='batch_seq_len')
        mask_idx_fw = tf.stack(
            [tf.range(tf.shape(input_layer)[0]), seq_len], axis=1)
        mask_idx_bw = tf.stack(
            [tf.range(tf.shape(input_layer)[0]), tf.zeros(shape=[tf.shape(input_layer)[0]], dtype=tf.int32)], axis=1)
        dropout_rate = tf.placeholder(dtype=tf.float32, name='dropout_rate')

    with tf.name_scope('RNN'):
        with tf.device('/GPU:0'):
            input_layer = tf.reshape(input_layer,
                                     shape=[tf.shape(input_layer)[0], tf.shape(input_layer)[1], 2048 * channel])
            cell_fw = make_cell(2048, tf.nn.selu, dropout_rate, mode)
            # cell_fw = tf.nn.rnn_cell.MultiRNNCell(
            #     cells=[make_cell(2048, tf.nn.selu, dropout_rate, mode) for _ in range(layer_num)],
            #     state_is_tuple=True)
            # cell_bw = tf.nn.rnn_cell.MultiRNNCell(
            #     cells=[make_cell(2048, tf.nn.selu, dropout_rate, mode) for _ in range(layer_num)],
            #     state_is_tuple=True)
            initial_state_fw = cell_fw.zero_state(tf.shape(input_layer)[0], tf.float32)
        with tf.device('/GPU:1'):
            out, state = tf.nn.dynamic_rnn(cell_fw,
                                           input_layer,
                                           sequence_length=None,
                                           initial_state=initial_state_fw,
                                           swap_memory=True,
                                           dtype=tf.float32,
                                           scope='rnn_networks')
        with tf.device('/GPU:0'):
            out_fw = tf.gather_nd(out, mask_idx_fw)
            # out_bw = tf.gather_nd(out, mask_idx_bw)

            # bn_fw = tf.layers.batch_normalization(out_fw, training=(mode == tf.estimator.ModeKeys.TRAIN), name='bn_fw')
            # bn_bw = tf.layers.batch_normalization(out_bw, training=(mode == tf.estimator.ModeKeys.TRAIN), name='bn_bw')
            # com = tf.multiply(out_fw, out_bw)
            # com = tf.nn.selu(com)

            fc = tf.layers.dense(inputs=out_fw, units=2048, activation=tf.nn.selu, name='fc')
            # fc_bn = tf.layers.batch_normalization(inputs=fc, training=(mode == tf.estimator.ModeKeys.TRAIN), name='fc_bn')
            dropoutfc = tf.layers.dropout(
                inputs=fc, rate=0.9, training=(mode == tf.estimator.ModeKeys.TRAIN), name='dropoutfc')
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
                "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }

            # decayed_learning_rate = learning_rate *
            #     decay_rate ^ (global_step / decay_steps)
            learning_rate = tf.train.exponential_decay(
                learning_rate=start_learning_rate,  # Base learning rate.
                global_step=epoche,  # Current index into the dataset.
                decay_steps=epech_decay,  # Decay step.
                decay_rate=decay_rate,  # Decay rate.
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
    return train_op, accuracy, loss, merged, out, state, out_fw


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


def load_npy_data(paths):
    return [np.load(p) for p in paths]


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


def data_batch_gen(data, label, path):
    paths = [os.path.join(path, d + '.npy') for d in data]

    data_queue, label_queue = tf.train.slice_input_producer(
        [tf.convert_to_tensor(paths, dtype=tf.string),
         tf.convert_to_tensor(label, dtype=tf.int32)],
        shuffle=False,
    )

    contents = tf.read_file(data_queue)
    contents = tf.decode_raw(contents, tf.float32)

    data_batch, label_batch = tf.train.batch(
        [contents, label_queue],
        batch_size=batch_size,
        dynamic_pad=False,
        allow_smaller_final_batch=True,
        enqueue_many=False,
        num_threads=1
    )
    return data_batch


def classify(train_list, test_list, image_feature_path, flow_feature_path_u, flow_feature_path_v):
    # train_list['data'] = train_list['data'][:2048]
    # test_list['data'] = test_list['data'][:2048]
    # train_list['label'] = train_list['label'][:2048]
    # test_list['label'] = test_list['label'][:2048]

    # remove data with frame number 1776
    del (train_list['data'][8151])
    del (train_list['label'][8151])

    train_list['data'] = np.array(train_list['data'])
    train_list['label'] = preprocessing.LabelEncoder().fit_transform(train_list['label'])
    train_list['label'] = np.array([ele + 1 for ele in train_list['label']])
    test_list['data'] = np.array(test_list['data'])
    test_list['label'] = preprocessing.LabelEncoder().fit_transform(test_list['label'])
    test_list['label'] = np.array([ele + 1 for ele in test_list['label']])

    # with tf.device('/GPU:1'):
    train_ops, accuracy, loss, summary, out, state, out_fw = rnn_time_model()

    global_steps = 1
    max_acc = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    config.allow_soft_placement = True
    with tf.device('/GPU:0'):
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('/home/boy2/UCF101/train', sess.graph)

        for i in range(int(total_epoch / train_epoch)):
            for j in range(train_epoch):
                # train
                train_list['data'], train_list['label'] = shuffle(train_list['data'], train_list['label'])
                train_acc = 0
                train_loss = 0
                total_train_batch = int(math.ceil(len(train_list['data']) / batch_size))
                for train_batch_num in range(total_train_batch):
                    batch_train_data_list = train_list['data'][
                                            batch_size * train_batch_num: batch_size * (train_batch_num + 1)]
                    batch_train_labels = train_list['label'][
                                         batch_size * train_batch_num: batch_size * (train_batch_num + 1)]
                    batch_train_image = load_data(batch_train_data_list, image_feature_path)
                    batch_train_flow_u = load_data(batch_train_data_list, flow_feature_path_u)
                    batch_train_flow_v = load_data(batch_train_data_list, flow_feature_path_v)
                    # pad 0s
                    batch_train_image, seq_len = padding(batch_train_image)
                    batch_train_flow_u, _ = padding(batch_train_flow_u, max_len=max(seq_len))
                    batch_train_flow_v, _ = padding(batch_train_flow_v, max_len=max(seq_len))

                    # batch_train_image, batch_train_labels, rgb_len = create_video_clips(batch_train_image,
                    #                                                                     batch_train_labels,
                    #                                                                     clip_len=clip_len)
                    # batch_train_flow_u = create_video_clips(batch_train_flow_u, rgb_len=rgb_len, clip_len=clip_len)
                    # batch_train_flow_v = create_video_clips(batch_train_flow_v, rgb_len=rgb_len, clip_len=clip_len)

                    batch_train_data = np.stack((batch_train_image, batch_train_flow_u, batch_train_flow_v), axis=-1)
                    # print(batch_train_data[0].shape)

                    _, loss_temp, accuracy_temp, summ_temp, out_temp, state_temp, out_fw_temp = sess.run(
                        [train_ops, loss, accuracy, summary, out, state, out_fw], feed_dict={
                            'inputs/input_layer:0': batch_train_data,
                            'inputs/labels:0': batch_train_labels,
                            'inputs/epoch:0': global_steps,
                            'inputs/batch_seq_len:0': seq_len,
                            # 'inputs/seq_len_size:0': len(seq_len),
                            'inputs/mode:0': tf.estimator.ModeKeys.TRAIN,
                            'inputs/dropout_rate:0': dropout
                        })
                    # print('TRAINING:', 'epoch:', i * train_epoch + j + 1, 'batch:', train_batch_num, 'loss:', loss_temp,
                    #       'accuracy', accuracy_temp)
                    train_acc += accuracy_temp
                    train_loss += loss_temp

                    sys.stdout.write('\r')
                    sys.stdout.write("[%-20s] %d%%" % ('=' * math.ceil(train_batch_num / (total_train_batch / 20)),
                                                       (100 / (total_train_batch)) * (train_batch_num + 1)))
                    sys.stdout.flush()
                print()
                print('________________________________________')
                print('Training for epoch', i * train_epoch + j + 1, 'the avg loss is', train_loss / train_batch_num,
                      'the avg acc is', train_acc / train_batch_num)
                global_steps += 1
            # validate
            eval_accuracy = 0
            eval_loss = 0
            total_eval_batch = int(math.ceil(len(test_list['data']) / batch_size))
            for eval_batch_num in range(total_eval_batch):
                batch_eval_data_list = test_list['data'][batch_size * eval_batch_num: batch_size * (eval_batch_num + 1)]
                # pad 0s
                batch_eval_image = load_data(batch_eval_data_list, image_feature_path)
                batch_eval_flow_u = load_data(batch_eval_data_list, flow_feature_path_u)
                batch_eval_flow_v = load_data(batch_eval_data_list, flow_feature_path_v)
                # pad 0s
                batch_eval_image, seq_len = padding(batch_eval_image)
                batch_eval_flow_u, _ = padding(batch_eval_flow_u, max_len=max(seq_len))
                batch_eval_flow_v, _ = padding(batch_eval_flow_v, max_len=max(seq_len))
                batch_eval_data = np.stack((batch_eval_image, batch_eval_flow_u, batch_eval_flow_v), axis=-1)
                # print(batch_eval_data[0].shape)
                batch_eval_labels = test_list['label'][batch_size * eval_batch_num: batch_size * (eval_batch_num + 1)]

                accuracy_temp, loss_temp = sess.run([accuracy, loss],
                                                    feed_dict={'inputs/input_layer:0': batch_eval_data,
                                                               'inputs/labels:0': batch_eval_labels,
                                                               'inputs/epoch:0': global_steps,
                                                               'inputs/batch_seq_len:0': seq_len,
                                                               # 'inputs/seq_len_size:0': len(seq_len),
                                                               'inputs/mode:0': tf.estimator.ModeKeys.PREDICT,
                                                               'inputs/dropout_rate:0': 0
                                                               })
                eval_accuracy += accuracy_temp
                eval_loss += loss_temp
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%%" % ('=' * math.ceil(eval_batch_num / (total_eval_batch / 20)),
                                                   (100 / (total_eval_batch)) * (eval_batch_num + 1)))
                sys.stdout.flush()
            eval_accuracy = eval_accuracy / eval_batch_num
            eval_loss = eval_loss / eval_batch_num
            print()
            print("_________________________________")
            print('EVALUATING: acc', eval_accuracy, 'loss:', eval_loss)
            if eval_accuracy < max_acc * 0.5:
                print(max_acc)
                return max_acc
            elif eval_accuracy > max_acc:
                max_acc = eval_accuracy
