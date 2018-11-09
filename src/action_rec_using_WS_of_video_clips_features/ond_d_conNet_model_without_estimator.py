import datetime
import math

import numpy as np
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

start_learning_rate = 0.000001
# 8 - 64
batch_size = 128
alpha = 0.2
num_unique_classes = 101
test_data_ratio = 0.3
total_epoch = 650
train_epoch = 1
epech_decay = math.ceil(9537 * 20 * train_epoch / batch_size)
decay_rate = 0.9
dropout_rate = 0.9
channel = 3
layer_num = 2
clip_len = 16
dim = 2
tra_data_splits = 1
eva_data_splits = 5
num_samples_per_test_video = 25


def _parse_fn(example):
    """Parse TFExample records and perform simple data augmentation."""
    example_fmt = {
        "feature": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(example, example_fmt)
    feature = tf.decode_raw(parsed["feature"], tf.float32)
    feature = tf.reshape(feature, [3, 2048, 2])
    # label = tf.decode_raw(parsed["labels"], tf.int64)
    return {'feature': feature}, parsed["label"]


def input_fn(train_input_path, eval_input_path):
    # train dataset
    train_dataset = tf.data.TFRecordDataset(train_input_path)
    train_dataset = train_dataset.shuffle(buffer_size=141993)
    train_dataset = train_dataset.map(map_func=_parse_fn)
    train_dataset = train_dataset.batch(batch_size=batch_size)
    train_dataset = train_dataset.repeat(count=None)

    # eval dataset
    eval_dataset = tf.data.TFRecordDataset(eval_input_path)
    eval_dataset = eval_dataset.map(map_func=_parse_fn)
    eval_dataset = eval_dataset.batch(batch_size=num_samples_per_test_video)
    eval_dataset = eval_dataset.repeat(count=None)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)

    # This is an op that gets the next element from the iterator
    data, label = iterator.get_next()
    # These ops let us switch and reinitialize every time we finish an epoch
    training_init_op = iterator.make_initializer(train_dataset)
    validation_init_op = iterator.make_initializer(eval_dataset)

    return data, label, training_init_op, validation_init_op


def nn_classifier(train_input_path, eval_input_path, params):
    with tf.device('/cpu:1'):
        data, labels, training_init_op, validation_init_op = input_fn(train_input_path, eval_input_path)
        inputlayer = tf.reshape(data['feature'], [-1, params['rgb_feature_shape'][0], params['rgb_feature_shape'][1],
                                                  params['rgb_feature_shape'][2]])
        inputlayer = tf.cast(inputlayer, tf.float32, name='input_rgb')
        inputlayer = tf.transpose(inputlayer, [0, 1, 3, 2])
        inputlayer = tf.reshape(inputlayer,
                                [-1, params['rgb_feature_shape'][0] * params['rgb_feature_shape'][2],
                                 params['rgb_feature_shape'][1], 1])
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
        mode = tf.placeholder(dtype=tf.bool, name='mode')

        conv1 = tf.layers.conv2d(inputs=inputlayer, filters=32,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 use_bias=True, kernel_size=[1, 1], padding="valid", activation=None,
                                 name='conv1')
        conv_bn1 = tf.layers.batch_normalization(conv1, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                 name='conv_bn1')
        conv_act1 = tf.nn.selu(conv_bn1, name="conv_act1")

        conv2 = tf.layers.conv2d(inputs=conv_act1, filters=16,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 use_bias=True, kernel_size=[1, 1], padding="valid", activation=None,
                                 name='conv2')
        conv_bn2 = tf.layers.batch_normalization(conv2, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                 name='conv_bn2')
        conv_act2 = tf.nn.selu(conv_bn2, name="conv_act2")

        conv3 = tf.layers.conv2d(inputs=conv_act2, filters=8,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 use_bias=True, kernel_size=[6, 1], padding="valid", activation=None,
                                 name='conv3')
        conv_bn3 = tf.layers.batch_normalization(conv3, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                 name='conv_bn3')
        conv_act3 = tf.nn.selu(conv_bn3, name="conv_act3")

        flat = tf.reshape(conv_act3, [-1, 1 * 2048 * 8], name='flat')

        fc1 = tf.layers.dense(inputs=flat, units=1024, activation=None, name='fc1')
        fc_bn1 = tf.layers.batch_normalization(inputs=fc1, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                               name='fc_bn1')
        fc_act1 = tf.nn.selu(fc_bn1, name="fc_act1")
        dropoutfc1 = tf.layers.dropout(
            inputs=fc_act1, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN), name='dropoutfc1')

        # fc2 = tf.layers.dense(inputs=dropoutfc1, units=1024, activation=None, name='fc2')
        # fc_bn2 = tf.layers.batch_normalization(inputs=fc2, training=(mode == tf.estimator.ModeKeys.TRAIN),
        #                                        name='fc_bn2')
        # fc_act2 = tf.nn.selu(fc_bn2, name="fc_act2")
        # dropoutfc2 = tf.layers.dropout(
        #     inputs=fc_act2, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN), name='dropoutfc2')
        #
        # fc3 = tf.layers.dense(inputs=dropoutfc2, units=1024, activation=None, name='fc3')
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

        _y, _idx, _count = tf.unique_with_counts(tf.argmax(logits, axis=1))

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "clip_accuracy": tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(input=logits, axis=1, output_type=tf.int64), labels), tf.float32),
                name="acc"),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "mean_softmax": tf.argmax(tf.reduce_mean(tf.nn.softmax(logits), axis=0, keep_dims=True), axis=1)[0],
            "max_vote": _y[tf.argmax(_count)]
        }

        # Configure the Training Op (for TRAIN mode)
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
            accuracy_softmax = tf.reduce_mean(
                tf.cast(tf.equal(predictions["mean_softmax"], labels[0]), tf.float32),
                name="acc")
            accuracy_logits = tf.reduce_mean(
                tf.cast(tf.equal(predictions["max_vote"], labels[0]), tf.float32),
                name="acc")
            eval_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels[0], logits=tf.reduce_mean(logits, axis=0, keep_dims=True)[0])
        return training_init_op, validation_init_op, train_op, loss, accuracy_softmax, accuracy_logits, predictions[
            "clip_accuracy"], summary_hook, predictions["mean_softmax"], predictions["max_vote"], labels[
                   0], eval_loss, logits


# def train_input_fn(filenames):
#     dataset = tf.data.TFRecordDataset(filenames)
#     dataset = dataset.shuffle(buffer_size=1024)
#     dataset = dataset.map(map_func=_parse_fn)
#     dataset = dataset.batch(batch_size=batch_size)
#     dataset = dataset.repeat(count=1)
#     iterator = dataset.make_one_shot_iterator()
#     features, labels = iterator.get_next()
#     return features, labels
#
#
# def test_input_fn(filenames):
#     dataset = tf.data.TFRecordDataset(filenames)
#     dataset = dataset.map(map_func=_parse_fn)
#     dataset = dataset.batch(batch_size=1)
#     dataset = dataset.repeat(count=1)
#     iterator = dataset.make_one_shot_iterator()
#     features, labels = iterator.get_next()
#     return features, labels


def classify(train_records, test_records, num_train_samples, num_test_samples, num_test_samples_per_video):
    global num_samples_per_test_video
    num_samples_per_test_video = num_test_samples_per_video

    # data, label, training_init_op, validation_init_op = input_fn(train_records, test_records)
    #
    # with tf.Session() as sess:
    #     sess.run(training_init_op)
    #     for j in range(1, int(math.ceil(num_train_samples / batch_size)) + 2):
    #         d, l = sess.run([data, label])
    #         if j == 1:
    #             _d = d
    #             _l = l
    #         print()

    with tf.device('/cpu:1'):
        params = {"rgb_feature_shape": (3, 2048, 2)}
        training_init_op, validation_init_op, train_op, loss, accuracy_softmax, \
        accuracy_logits, accuracy_clips, summary_hook, pred_softmax, pred_max_vote, pred_true, eval_loss, logits \
            = nn_classifier(train_records, test_records, params)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        learning_rate = start_learning_rate
        best_result = 0
        prev_loss = 10

        exp_result = "/home/boy2/UCF101/ucf101_dataset/exp_results/res_for_1dconv_classifier_at_" + str(
            datetime.datetime.now())
        print("Training start____________")
        for i in range(1, total_epoch + 1):
            # training
            sess.run(training_init_op)
            mode = tf.estimator.ModeKeys.TRAIN
            total_train_loss = 0
            train_loss = 0
            # 269894 the # of training samples
            for j in range(1, int(math.ceil(num_train_samples / batch_size)) * train_epoch + 1):
                _, loss_temp = sess.run([train_op, loss], feed_dict={'learning_rate:0': learning_rate, 'mode:0': mode})
                total_train_loss += loss_temp
                train_loss += loss_temp
                if j % 100 == 0:
                    print("Setp", j, "The loss is", train_loss / 100)
                    train_loss = 0
            print("Training epoch", i * train_epoch, "finished, the avg loss is", total_train_loss / j)

            # evaluation
            sess.run(validation_init_op)
            mode = tf.estimator.ModeKeys.EVAL
            eval_acc_softmax = 0
            eval_acc_logits = 0
            eval_acc_clips = 0
            eval_loss_clips = 0
            eval_loss_softmax = 0
            pre_soft = []
            pre_vote = []
            true_labels = []
            print("______EVALUATION________")
            # 105164 the # for testing samples
            for j in range(1, int(math.ceil(num_test_samples / num_samples_per_test_video)) + 1):
                loss_temp, accuracy_softmax_temp, accuracy_logits_temp, accuracy_clips_temp, pred_softmax_temp, \
                pred_max_vote_temp, pred_true_temp, eval_loss_temp = sess.run(
                    [loss, accuracy_softmax, accuracy_logits, accuracy_clips, pred_softmax, pred_max_vote, pred_true,
                     eval_loss],
                    feed_dict={'learning_rate:0': learning_rate, 'mode:0': mode})
                eval_acc_softmax += accuracy_softmax_temp
                eval_acc_logits += accuracy_logits_temp
                eval_acc_clips += accuracy_clips_temp
                eval_loss_clips += loss_temp
                eval_loss_softmax += eval_loss_temp
                pre_soft.append(pred_softmax_temp)
                pre_vote.append(pred_max_vote_temp)
                true_labels.append(pred_true_temp)
            # cm_softmax = sess.run(
            #     tf.confusion_matrix(labels=true_labels, predictions=pre_soft, num_classes=num_unique_classes))
            # cm_vote = sess.run(
            #     tf.confusion_matrix(labels=true_labels, predictions=pre_vote, num_classes=num_unique_classes))
            eval_acc_softmax /= j
            eval_acc_clips /= j
            eval_acc_logits /= j
            eval_loss_clips /= j
            eval_loss_softmax /= j
            print("Accuracy softmax for evaluation is:", eval_acc_softmax, "\n",
                  "Accuracy logits for evaluation is:", eval_acc_logits, "\n",
                  "Accuracy clips for evaluation is:", eval_acc_clips, "\n",
                  "loss for clips is", eval_loss_clips, "loss for softmax is", eval_loss_softmax)
            eval_acc = max([eval_acc_softmax, eval_acc_logits, eval_acc_clips])
            evaluation_loss = min([eval_loss_clips, eval_loss_softmax])

            with open(exp_result, "a") as text_file:
                text_file.writelines(
                    "Evaluation accuracy after training epoch %s is: %s \n" % (i * train_epoch, eval_acc))
                # text_file.writelines(
                #     "Softmax confusion matrix after training epoch %s is: \n" % (i * train_epoch))
                # np.savetxt(text_file, cm_softmax, fmt='%s')
                # text_file.writelines(
                #     "Vote confusion matrix after training epoch %s is: \n" % (i * train_epoch,))
                # np.savetxt(text_file, cm_vote, fmt='%s')
            if eval_acc > best_result:
                best_result = eval_acc
            if evaluation_loss > prev_loss + 0.01:
                learning_rate *= 0.1
                print('The learning rate is decreased to', learning_rate)
            prev_loss = evaluation_loss
            print("_________EVALUATION DONE___________")

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
