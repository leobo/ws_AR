import datetime
import math

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

start_learning_rate = 0.0001
batch_size = 64
num_unique_classes = 101
total_epoch = 1000
train_epoch = 3
dropout_rate = 0.8
num_samples_per_test_video = 1


def _parse_fn(example):
    """Parse TFExample records and perform simple data augmentation."""
    example_fmt = {
        "features": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(example, example_fmt)
    features = tf.decode_raw(parsed["features"], tf.float32)
    features = tf.reshape(features, feature_size)
    return {'features': features}, parsed["label"]


def input_fn(train_input_path, eval_input_path):
    # train dataset
    train_dataset = tf.data.TFRecordDataset(train_input_path)
    train_dataset = train_dataset.shuffle(buffer_size=9537)
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
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
        mode = tf.placeholder(dtype=tf.bool, name='mode')

        # input_data = tf.reshape(data['features'], [params['feature_shape'][0], params['feature_shape'][1],
        #                                            params['feature_shape'][2] * params['feature_shape'][3], 1])
        input_data = tf.transpose(tf.reshape(data['features'], params['feature_shape']), [0, 1, 3, 2])
        input_data_1 = input_data[:, :params["dim"], :, :]
        input_data_2 = input_data[:, params["dim"]:, :, :]

        # model 1
        conv1_m1 = tf.layers.conv2d(inputs=input_data_1, filters=2048,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    use_bias=True, kernel_size=[3, 3], padding="valid", activation=None, strides=[1, 1],
                                    name='conv1_m1')
        conv_bn1_m1 = tf.layers.batch_normalization(conv1_m1, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                    name='conv_bn1_m1')
        conv_act1_m1 = tf.nn.relu(conv_bn1_m1, name="conv_act1_m1")

        conv1_m2 = tf.layers.conv2d(inputs=input_data_2, filters=2048,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    use_bias=True, kernel_size=[3, 3], padding="valid", activation=None,
                                    strides=[1, 1],
                                    name='conv1_m2')
        conv_bn1_m2 = tf.layers.batch_normalization(conv1_m2, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                    name='conv_bn1_m2')
        conv_act1_m2 = tf.nn.relu(conv_bn1_m2, name="conv_act1_m2")

        mid = tf.concat([conv_act1_m1, conv_act1_m2], axis=-1)

    with tf.device('/gpu:1'):
        conv3 = tf.layers.conv2d(inputs=mid, filters=2048, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 use_bias=False, kernel_size=[1, 1], padding="same", activation=None, strides=[1, 1],
                                 name='conv3')
        conv_bn3 = tf.layers.batch_normalization(conv3, training=(mode == tf.estimator.ModeKeys.TRAIN), name='conv_bn3')
        conv_act3 = tf.nn.relu(conv_bn3, name="conv_act3")
        dropoutfc1 = tf.layers.dropout(
            inputs=conv_act3, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN), name='dropoutfc1')

        # conv4 = tf.layers.conv2d(inputs=dropoutfc1, filters=2048,
        #                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                          use_bias=False, kernel_size=[1, 1], padding="same", activation=None, strides=[1, 1],
        #                          name='conv4')
        # conv_bn4 = tf.layers.batch_normalization(conv4, training=(mode == tf.estimator.ModeKeys.TRAIN), name='conv_bn4')
        # conv_act4 = tf.nn.relu(conv_bn4, name="conv_act4")
        # dropoutfc2 = tf.layers.dropout(
        #     inputs=conv_act4, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN), name='dropoutfc2')

        logits = tf.layers.conv2d(
            inputs=dropoutfc1,
            filters=num_unique_classes,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            use_bias=False, kernel_size=[1, 1], padding="same", activation=None, strides=[1, 1],
            name='logits')

        logits = tf.reshape(logits, [-1, num_unique_classes])
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=num_unique_classes)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

        # _y, _idx, _count = tf.unique_with_counts(tf.argmax(logits, axis=1))

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "clip_accuracy": tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(input=logits, axis=1, output_type=tf.int64), labels), tf.float32),
                name="acc")
        }

        # Configure the Training Op (for TRAIN mode)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step)
        tf.summary.scalar('loss', loss)
        summary_hook = tf.train.SummarySaverHook(
            save_steps=100, output_dir='/tmp/tf', summary_op=tf.summary.merge_all()
        )

        return training_init_op, validation_init_op, train_op, loss, predictions[
            "clip_accuracy"], summary_hook, labels, logits


def classify(train_records, test_records, num_train_samples, num_test_samples, num_test_samples_per_video,
             feature_shape, dim):
    global num_samples_per_test_video
    num_samples_per_test_video = num_test_samples_per_video
    global feature_size
    feature_size = feature_shape

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
    with tf.device('/cpu:1'):
        params = {"feature_shape": (-1, feature_size[0], feature_size[1], feature_size[2]), "dim": dim}
        global_step = tf.Variable(0, name="global_step", trainable=False)
        training_init_op, validation_init_op, train_op, loss, accuracy_clips, summary_hook, \
        pred_true, logits = nn_classifier(train_records, test_records, global_step, params)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.import_meta_graph('/home/boy2/UCF101/ucf101_dataset/frame_features/checkpoint/model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('/home/boy2/UCF101/ucf101_dataset/frame_features/checkpoint/'))

        learning_rate = start_learning_rate
        best_result = 0
        prev_acc = 0

        exp_result = "/home/boy2/UCF101/ucf101_dataset/exp_results/res_for_1dconv_classifier_at_" + str(
            datetime.datetime.now())
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
            true_labels = []
            print("______EVALUATION________")
            # 105164 the # for testing samples
            for j in range(1, int(math.ceil(num_test_samples / batch_size)) + 1):
                # print(int(math.ceil(num_test_samples / batch_size)))
                # print("In test:", j)
                loss_temp, accuracy_clips_temp, pred_true_temp = sess.run(
                    [loss, accuracy_clips, pred_true],
                    feed_dict={'learning_rate:0': learning_rate, 'mode:0': mode})
                eval_acc_clips += accuracy_clips_temp
                eval_loss_clips += loss_temp
                true_labels.append(pred_true_temp)
            eval_acc_clips /= j
            eval_loss_clips /= j
            print("Accuracy clips for evaluation is:", eval_acc_clips, "\n",
                  "loss for clips is", eval_loss_clips)
            eval_acc = eval_acc_clips
            with open(exp_result, "a") as text_file:
                text_file.writelines(
                    "Evaluation accuracy after training epoch %s is: %s \n" % (i * train_epoch, eval_acc))
                text_file.writelines(
                    "Softmax confusion matrix after training epoch %s is: \n" % (i * train_epoch))
                text_file.writelines(
                    "Vote confusion matrix after training epoch %s is: \n" % (i * train_epoch,))
            if eval_acc < prev_acc:
                learning_rate *= 0.1
                print('The learning rate is decreased to', learning_rate)
            if eval_acc > best_result:
                best_result = eval_acc
            prev_acc = eval_acc
            print("_________EVALUATION DONE___________")
            saver.save(sess, "/home/boy2/UCF101/ucf101_dataset/frame_features/checkpoint/model.ckpt")
        return best_result
