import datetime
import math

import numpy as np
import sklearn
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

start_learning_rate = 0.0001
# 8 - 64
batch_size = 128
alpha = 0.2
num_unique_classes = 101
test_data_ratio = 0.3
total_epoch = 120
train_epoch = 1
epech_decay = math.ceil(9537 * 20 * train_epoch / batch_size)
decay_rate = 0.9
dropout_rate = 0.95
channel = 3
layer_num = 2
clip_len = 16
dim = 2
tra_data_splits = 1
eva_data_splits = 5
num_samples_per_test_video = 25
rgb_size = (1, 2048, 2)
flow_size = (2, 2048, 2)


def test_accuracy(gt, predicts, num_test_samples, num_test_samples_per_video, num_trans_m):
    gt = np.concatenate(gt, axis=0)
    gt = np.reshape(gt,
                    [(num_test_samples_per_video*num_trans_m), int(num_test_samples / (num_test_samples_per_video*num_trans_m))])
    gt = np.mean(gt, axis=0, dtype=np.int64, keepdims=False)
    predicts = np.concatenate(predicts, axis=0)
    predicts = np.reshape(predicts,
                          [(num_test_samples_per_video*num_trans_m), int(num_test_samples / (num_test_samples_per_video*num_trans_m)),
                           num_unique_classes])
    pre_softmax = np.argmax(np.mean(predicts, axis=0, keepdims=False), axis=-1)
    return sklearn.metrics.accuracy_score(gt, pre_softmax)


def _parse_fn(example):
    """Parse TFExample records and perform simple data augmentation."""
    example_fmt = {
        "rgb": tf.FixedLenFeature([], tf.string),
        "flow": tf.FixedLenFeature([], tf.string),
        "labels": tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(example, example_fmt)
    rgb = tf.decode_raw(parsed["rgb"], tf.float32)
    rgb = tf.reshape(rgb, rgb_size)
    flow = tf.decode_raw(parsed["flow"], tf.float32)
    flow = tf.reshape(flow, flow_size)
    # label = tf.decode_raw(parsed["labels"], tf.int64)
    return {'rgb': rgb, 'flow': flow}, parsed["labels"]


def input_fn(train_input_path, eval_input_path):
    # train dataset
    train_dataset = tf.data.TFRecordDataset(train_input_path)
    train_dataset = train_dataset.shuffle(buffer_size=95370)
    train_dataset = train_dataset.map(map_func=_parse_fn)
    train_dataset = train_dataset.batch(batch_size=batch_size)
    train_dataset = train_dataset.repeat(count=None)

    # eval dataset
    eval_dataset = tf.data.TFRecordDataset(eval_input_path)
    eval_dataset = eval_dataset.map(map_func=_parse_fn)
    eval_dataset = eval_dataset.batch(batch_size=batch_size)
    eval_dataset = eval_dataset.repeat(count=None)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)

    # This is an op that gets the next element from the iterator
    data, label = iterator.get_next()
    # These ops let us switch and reinitialize every time we finish an epoch
    training_init_op = iterator.make_initializer(train_dataset)
    validation_init_op = iterator.make_initializer(eval_dataset)

    return data, label, training_init_op, validation_init_op


def nn_classifier(train_input_path, eval_input_path, global_step, params):
    with tf.device('/gpu:0'):
        data, labels, training_init_op, validation_init_op = input_fn(train_input_path, eval_input_path)
    with tf.device('/gpu:1'):
        input_rgb = tf.reshape(data['rgb'], [-1, params['rgb_feature_shape'][0], params['rgb_feature_shape'][1],
                                             params['rgb_feature_shape'][2]])
        input_rgb = tf.transpose(input_rgb, [0, 3, 1, 2])
        # input_rgb = tf.subtract(input_rgb, tf.constant(params['max_value'][0]))
        input_rgb = tf.divide(input_rgb, tf.constant(params['max_value'][0]))
        input_flow = tf.reshape(data['flow'],
                                [-1, params['flow_feature_shape'][0], params['flow_feature_shape'][1],
                                 params['flow_feature_shape'][2]])
        input_flow = tf.transpose(input_flow, [0, 3, 1, 2])
        # input_flow = tf.subtract(input_flow, tf.constant(params['max_value'][1:]))
        input_flow = tf.divide(input_flow, tf.constant(params['max_value'][1:]))
        input_rgb = tf.cast(input_rgb, tf.float32, name='input_rgb')
        input_flow = tf.cast(input_flow, tf.float32, name='input_flow')

        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
        mode = tf.placeholder(dtype=tf.bool, name='mode')

        # input_data = tf.concat([input_rgb, tf.reshape(input_flow[:, 0, :, :], [-1, 1, params['flow_feature_shape'][1],
        #                                                                        params['flow_feature_shape'][2]]),
        #                         tf.reshape(input_flow[:, 1, :, :], [-1, 1, params['flow_feature_shape'][1],
        #                                                             params['flow_feature_shape'][2]])], axis=1)
        # input_data = tf.transpose(tf.concat([input_rgb, input_flow], axis=2), [0, 2, 1, 3])
        input_data = tf.reshape(tf.concat([input_rgb, input_flow], axis=2),
                                [-1, params['flow_feature_shape'][2], 2048 * 3, 1])
        # input_data = tf.reshape(input_data, [-1, 4, 4, 2048*3])

        # input_data = tf.concat([input_rgb, input_flow], axis=1)
        # input_data = tf.transpose(input_data, [0,1,3,2])

        conv1 = tf.layers.conv2d(inputs=input_data, filters=8,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 use_bias=True, kernel_size=[1, 1], padding="same", activation=None,
                                 strides=[1, 1],
                                 name='conv1')
        conv_bn1 = tf.layers.batch_normalization(conv1, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                 name='conv_bn1')
        conv_act1 = tf.nn.relu(conv_bn1, name="conv_act1")
        # pool1 = tf.layers.max_pooling2d(inputs=conv_act1, pool_size=[2, 1], strides=[2, 1], padding='valid')

        conv2 = tf.layers.conv2d(inputs=conv_act1, filters=4,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 use_bias=True, kernel_size=[8, 1], padding="same",
                                 activation=None, strides=[8, 1],
                                 name='conv2')
        conv_bn2 = tf.layers.batch_normalization(conv2, training=(mode == tf.estimator.ModeKeys.TRAIN), name='conv_bn2')
        conv_act2 = tf.nn.relu(conv_bn2, name="conv_act2")

        # pool2 = tf.layers.max_pooling2d(inputs=conv_act2, pool_size=[2, 1], strides=[2, 1], padding='valid')

        # conv3 = tf.layers.conv2d(
        #     inputs=pool2,
        #     filters=4,
        #     kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #     use_bias=False, kernel_size=[8, 1], padding="valid", activation=None, strides=[1, 1],
        #     name='conv3')
        # conv_bn3 = tf.layers.batch_normalization(conv3, training=(mode == tf.estimator.ModeKeys.TRAIN), name='conv_bn3')
        # conv_act3 = tf.nn.relu(conv_bn3, name="conv_act3")

        # conv4 = tf.layers.conv2d(inputs=conv_act3, filters=2048,
        #                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                          use_bias=False, kernel_size=[1, 2], padding="same", activation=None, strides=[1, 1],
        #                          name='conv4')
        # conv_bn4 = tf.layers.batch_normalization(conv4, training=(mode == tf.estimator.ModeKeys.TRAIN), name='conv_bn4')
        # conv_act4 = tf.nn.relu(conv_bn4, name="conv_act4")
        # pool4 = tf.layers.average_pooling2d(inputs=conv_act4, pool_size=[1, 2], strides=[1, 1])
        #
        # conv5 = tf.layers.conv2d(
        #     inputs=tf.transpose(tf.reshape(input_flow[:, 1, :, :], [-1, 1, params['flow_feature_shape'][1],
        #                                                             params['flow_feature_shape'][2]]), [0, 1, 3, 2]),
        #     filters=2048,
        #     kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #     use_bias=False, kernel_size=[1, 1], padding="same", activation=None, strides=[1, 1],
        #     name='conv5')
        # conv_bn5 = tf.layers.batch_normalization(conv5, training=(mode == tf.estimator.ModeKeys.TRAIN), name='conv_bn5')
        # conv_act5 = tf.nn.relu(conv_bn5, name="conv_act5")
        #
        # conv6 = tf.layers.conv2d(inputs=conv_act5, filters=2048,
        #                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                          use_bias=False, kernel_size=[1, 2], padding="same", activation=None, strides=[1, 1],
        #                          name='conv6')
        # conv_bn6 = tf.layers.batch_normalization(conv6, training=(mode == tf.estimator.ModeKeys.TRAIN), name='conv_bn6')
        # conv_act6 = tf.nn.relu(conv_bn6, name="conv_act6")
        # pool6 = tf.layers.average_pooling2d(inputs=conv_act6, pool_size=[1, 2], strides=[1, 1])

        # conv_act3 = tf.reduce_mean(input_data, axis=1)

        flat_rgb = tf.reshape(conv_act2, [-1, 3 * 2048 * 4], name='flat_rgb')

        # flat_u = tf.reshape(pool4, [-1, 1 * 2048 * 1 * 1], name='flat_u')
        # flat_v = tf.reshape(pool6, [-1, 1 * 2048 * 1 * 1], name='flat_v')

        with tf.device('/gpu:0'):
            fc1 = tf.layers.dense(inputs=flat_rgb, units=4096, activation=None, use_bias=True, name='fc1')
            fc_bn1 = tf.layers.batch_normalization(inputs=fc1, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                   name='fc_bn1')
            fc_act1 = tf.nn.relu(fc_bn1, name="fc_act1")
            dropoutfc1 = tf.layers.dropout(
                inputs=fc_act1, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN),
                name='dropoutfc1')

            fc2 = tf.layers.dense(inputs=dropoutfc1, units=4096, activation=None, use_bias=True, name='fc2')
            fc_bn2 = tf.layers.batch_normalization(inputs=fc2, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                   name='fc_bn2')
            fc_act2 = tf.nn.relu(fc_bn2, name="fc_act2")
            dropoutfc2 = tf.layers.dropout(
                inputs=fc_act2, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN),
                name='dropoutfc2')

            # fc3 = tf.layers.dense(inputs=flat_v, units=1024, activation=None, name='fc3')
            # fc_bn3 = tf.layers.batch_normalization(inputs=fc3, training=(mode == tf.estimator.ModeKeys.TRAIN),
            #                                        name='fc_bn3')
            # fc_act3 = tf.nn.selu(fc_bn3, name="fc_act3")
            # dropoutfc3 = tf.layers.dropout(
            #     inputs=fc_act3, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN), name='dropoutfc3')

        logits = tf.layers.dense(inputs=dropoutfc2, units=num_unique_classes, use_bias=False, name='logits_rgb')
        # logits_u = tf.layers.dense(inputs=dropoutfc2, units=num_unique_classes, name='logits_u')
        # logits_v = tf.layers.dense(inputs=dropoutfc3, units=num_unique_classes, name='logits_v')
        # logits = tf.reduce_mean(tf.stack([logits_rgb, logits_u, logits_v], axis=1), axis=1)

        # Calculate Loss (for both TRAIN and EVAL modes)
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=num_unique_classes)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "clip_accuracy": tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(input=logits, axis=1, output_type=tf.int64), labels), tf.float32),
                name="acc"),
        }

        # Configure the Training Op (for TRAIN mode)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=1, total_num_replicas=None)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step)
        tf.summary.scalar('loss', loss)
        summary_hook = tf.train.SummarySaverHook(
            save_steps=100, output_dir='/tmp/tf', summary_op=tf.summary.merge_all()
        )

        return training_init_op, validation_init_op, train_op, loss, predictions[
            "clip_accuracy"], summary_hook, logits, tf.nn.softmax(logits), labels


def classify(train_records, test_records, num_train_samples, num_test_samples, num_test_samples_per_video, num_trans_m,
             rgb_shape, flow_shape, max_value):
    global num_samples_per_test_video
    num_samples_per_test_video = num_test_samples_per_video
    global rgb_size
    rgb_size = rgb_shape
    global flow_size
    flow_size = flow_shape

    # data, label, training_init_op, validation_init_op = input_fn(train_records, test_records)
    # with tf.Session() as sess:
    #     sess.run(validation_init_op)
    #     for j in range(1, 251):
    #         d, l = sess.run([data, label])
    #         if len(l) != batch_size:
    #             _d = d
    #             _l = l
    #         print()

    tf.reset_default_graph()
    with tf.device('/gpu:1'):
        params = {"rgb_feature_shape": rgb_size,
                  "flow_feature_shape": flow_size,
                  "max_value": np.load(max_value)}
        global_step = tf.Variable(0, name="global_step", trainable=False)
        training_init_op, validation_init_op, train_op, loss, accuracy_clips, summary_hook, logits, softmax, labels \
            = nn_classifier(train_records, test_records, global_step, params)
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        # saver = tf.train.import_meta_graph('/home/boy2/ucf101/ucf101_dataset/frame_features/checkpoint/model.ckpt.meta')
        # saver.restore(sess, tf.train.latest_checkpoint('/home/boy2/ucf101/ucf101_dataset/frame_features/checkpoint/'))

        learning_rate = start_learning_rate
        best_result = 0
        prev_loss = 10
        prev_acc = 0

        exp_result = "/home/boy2/ucf101/ucf101_dataset/exp_results/res_for_1dconv_classifier_at_" + str(
            datetime.datetime.now()) + '.txt'
        print("Training start____________")
        for i in range(0, total_epoch + 1, train_epoch):
            # training
            sess.run(training_init_op)
            mode = tf.estimator.ModeKeys.TRAIN
            total_train_loss = 0
            train_loss = 0
            # 269894 the # of training samples
            for j in range(1, int(math.ceil(num_train_samples / batch_size)) * train_epoch + 1):
                _, loss_temp = sess.run([train_op, loss],
                                        feed_dict={'learning_rate:0': learning_rate, 'mode:0': mode})
                total_train_loss += loss_temp
                train_loss += loss_temp
                if j % 100 == 0:
                    print("Setp", j, "The loss is", train_loss / 100)
                    train_loss = 0
            print("Training epoch", i, "finished, the avg loss is",
                  total_train_loss / j)

            # evaluation
            sess.run(validation_init_op)
            mode = tf.estimator.ModeKeys.EVAL
            eval_acc_clips = 0
            eval_loss_clips = 0
            pre_softmax = []
            pre_logits = []
            gt = []
            print("______EVALUATION________")
            # 105164 the # for testing samples
            for j in range(1, int(math.ceil(num_test_samples / batch_size)) + 1):
                b = int(math.ceil(num_test_samples / batch_size)) + 1
                loss_temp, accuracy_clips_temp, logits_temp, softmax_temp, labels_temp = sess.run(
                    [loss, accuracy_clips, logits, softmax, labels],
                    feed_dict={'learning_rate:0': learning_rate, 'mode:0': mode})
                eval_acc_clips += accuracy_clips_temp
                eval_loss_clips += loss_temp
                pre_softmax.append(softmax_temp)
                pre_logits.append(logits_temp)
                gt.append(labels_temp)
            eval_acc_clips /= j
            eval_loss_clips /= j
            # calculate the accuracies based on softmax_mean and logits_vote
            eval_acc_softmax = test_accuracy(gt, pre_softmax, num_test_samples, num_test_samples_per_video, num_trans_m)
            eval_acc_logits = test_accuracy(gt, pre_logits, num_test_samples, num_test_samples_per_video, num_trans_m)

            print("Accuracy softmax for evaluation is:", eval_acc_softmax, "\n",
                  "Accuracy logits for evaluation is:", eval_acc_logits, "\n",
                  "Accuracy clips for evaluation is:", eval_acc_clips, "\n",
                  "loss is", eval_loss_clips)
            eval_acc = min([eval_acc_softmax, eval_acc_logits, eval_acc_clips])
            evaluation_loss = eval_loss_clips

            with open(exp_result, "a") as text_file:
                text_file.writelines(
                    "Evaluation accuracy after training epoch %s is: %s \n" % (i * train_epoch, eval_acc))
                # text_file.writelines(
                #     "Softmax confusion matrix after training epoch %s is: \n" % (i * train_epoch))
                # np.savetxt(text_file, cm_softmax, fmt='%s')
                # text_file.writelines(
                #     "Vote confusion matrix after training epoch %s is: \n" % (i * train_epoch,))
                # np.savetxt(text_file, cm_logits, fmt='%s')
            if eval_acc < prev_acc:
                # best_result = eval_acc
                # if evaluation_loss > prev_loss + 0.01:
                #     first = 0
                learning_rate *= 0.1
                print('The learning rate is decreased to', learning_rate)
            # if learning_rate <= 0.0000001:
            #     break
            if max([eval_acc_softmax, eval_acc_logits, eval_acc_clips]) > best_result:
                best_result = max([eval_acc_softmax, eval_acc_logits, eval_acc_clips])
            prev_loss = evaluation_loss
            prev_acc = eval_acc
            print("_________EVALUATION DONE___________")
            saver.save(sess, "/home/boy2/ucf101/ucf101_dataset/model_check/check_point")
        return best_result

    # for _ in range(1, total_epoch):
    #     mnist_classifier.train(
    #         input_fn=train_input_fuc,
    #         steps=None,
    #         # steps=int(9537 * 4 * train_epoch / batch_size),
    #         hooks=[logging_hook])
    #
    #     # evaluation
    #     print("______EVALUATION________")
    #     eval = mnist_classifier.evaluate(input_fn=test_input_fnc)
    #     eval_acc = eval['accuracy']
    #     with open(exp_result, "a") as text_file:
    #         text_file.writelines(
    #             "Evaluation accuracy after training epoch %s is: %s \n" % (_ * train_epoch, eval_acc))
    #     print("Accuracy for evaluation is:", eval_acc)
    #     if eval_acc < best_result * 0.5:
    #         print(best_result)
    #         # return best_result
    #     elif eval_acc > best_result:
    #         best_result = eval_acc
    #     print("______EVALUATION DONE________")

    # print("_________EVALUATION START___________")
    # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"rgb": np.array(test_rgb), "flow": np.array(test_flow),
    #        "lr": np.array([learning_rate for _ in range(len(test_rgb))])},
    #     y=test_labels,
    #     batch_size=25,
    #     num_epochs=1,
    #     shuffle=False,
    #     num_threads=1
    # )
    # eval = mnist_classifier.evaluate(input_fn=test_input_fnc)
    # with open(exp_result, "a") as text_file:
    #     text_file.writelines(
    #         "Evaluation accuracy after training epoch %s is: %s \n" % (_ * train_epoch, eval['accuracy']))
    # if eval['accuracy'] >= prev_result:
    #     if eval['accuracy'] > best_result:
    #         best_result = eval['accuracy']
    # else:
    #     if eval['accuracy'] < (prev_result * 0.5):
    #         print("Training will stop, the best result is", best_result)
    #         # return best_result
    #     elif (prev_result * 0.5) <= eval['accuracy'] < (prev_result - 0.01) or eval["accuracy"] < \
    #             best_result - 0.05:
    #         learning_rate *= 0.1
    #         print('The learning rate is decreased to', learning_rate)
    #     elif eval["accuracy"] > 0.84 and first84 is True:
    #         learning_rate *= 0.1
    #         print('The learning rate is decreased to', learning_rate)
    #         first84 = False
    #     elif eval["accuracy"] > 0.845 and first845 is True:
    #         learning_rate *= 0.5
    #         print('The learning rate is decreased to', learning_rate)
    #         first845 = False
    #     elif eval["accuracy"] > 0.85 and first85 is True:
    #         learning_rate *= 0.1
    #         print('The learning rate is decreased to', learning_rate)
    #         first85 = False
    #     elif eval["accuracy"] > 0.852 and first853 is True:
    #         learning_rate *= 0.5
    #         print('The learning rate is decreased to', learning_rate)
    #         first853 = False
    #     elif eval["accuracy"] > 0.86 and first86 is True:
    #         learning_rate *= 0.1
    #         print('The learning rate is decreased to', learning_rate)
    #         first86 = False
    # prev_result = eval['accuracy']
    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"rgb": np.array(train_rgb), "flow": np.array(train_flow),
    #        "lr": np.array([learning_rate for _ in range(len(train_rgb))])},
    #     y=train_labels,
    #     batch_size=batch_size,
    #     num_epochs=train_epoch,
    #     shuffle=True,
    #     num_threads=1
    # )
    # print("_________EVALUATION DONE___________")
