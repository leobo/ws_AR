from npyFileReader import Npyfilereader
from models.research.slim.nets import resnet_v1, vgg

import os

import numpy as np
import tensorflow as tf
from sklearn import preprocessing

from classifier import twodconv_classifier, nn_classifier, onedconv_classifier
from trainTestSamplesGen import TrainTestSampleGen
from weighted_sum import calWeightedSum

from npyFileReader import Npyfilereader
from weighted_sum.videoDescriptorWeightedSum import Weightedsum
from preprocessing.cropper import Cropper


slim = tf.contrib.slim

start_learning_rate = 0.0005
# 8 - 64
batch_size = 64
dropout_rate = 0.9
alpha = 0.2
num_unique_classes = 101
test_data_ratio = 0.3
epech_decay = 20
decay_rate = 0.9
total_epoch = 600
train_epoch = 10

width = 224
height = 224
color_channels = 3

def extract(train, test):
    input_layer = tf.placeholder(dtype=tf.float32, shape=[None, width, height, color_channels])
    with tf.Session() as sess:
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            resNet152, end_points = resnet_v1.resnet_v1_152(input_layer,
                                                            num_classes=None,
                                                            is_training=False,
                                                            global_pool=True,
                                                            output_stride=None,
                                                            spatial_squeeze=True,
                                                            reuse=tf.AUTO_REUSE,
                                                            scope='resnet_v1_152')
        saver = tf.train.Saver()
        saver.restore(sess, "/home/boy2/UCF101/src/resNet-152/resnet_v1_152.ckpt")

        train_feature = []
        num_chunks = np.ceil(len(train) / 100)
        chunks = np.array_split(np.array(train), num_chunks)
        for c in chunks:
            # make input tensor for resNet152
            train_feature += list(np.reshape(sess.run(resNet152, feed_dict={input_layer: c}),
                                       newshape=[len(c), 2048]))
        train_feature = np.array(train_feature)

        test_feature = []
        num_chunks = np.ceil(len(test) / 100)
        chunks = np.array_split(np.array(test), num_chunks)
        for c in chunks:
            # make input tensor for resNet152
            test_feature += list(np.reshape(sess.run(resNet152, feed_dict={input_layer: c}),
                                             newshape=[len(c), 2048]))
        test_feature = np.array(test_feature)
    return train_feature, test_feature

def model(features, labels, mode, params):
    # resNet_saver = tf.train.import_meta_graph("/home/boy2/UCF101/src/resNet-152/temp.meta")
    # resNet_graph = tf.get_default_graph()
    # resNet_end = resNet_graph.get_tensor_by_name('resnet_v1_152/block4/unit_3/bottleneck_v1/Relu:0')
    # resNet_end_pool = tf.layers.max_pooling2d(resNet_end, pool_size=(7, 7), strides=1, name='resNet_end_pool')
    # flat = tf.reshape(resNet_end_pool, shape=(-1, 2048), name='flat')


    input_layer = tf.reshape(features['x'], [-1, 3, 2048, 1])
    input_layer = tf.cast(input_layer, tf.float32, name='input_layer')

    # vgg19, end_points = vgg.vgg_19(input_layer,
    #                                num_classes=None,
    #                                is_training=(mode == tf.estimator.ModeKeys.TRAIN),
    #                                dropout_keep_prob=0.5,
    #                                spatial_squeeze=True,
    #                                scope='vgg_19',
    #                                fc_conv_padding='VALID',
    #                                global_pool=False)

    # with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    #     resNet152, end_points = resnet_v1.resnet_v1_152(input_layer,
    #                                                     num_classes=None,
    #                                                     is_training=(mode == tf.estimator.ModeKeys.TRAIN),
    #                                                     global_pool=True,
    #                                                     output_stride=None,
    #                                                     spatial_squeeze=True,
    #                                                     reuse=tf.AUTO_REUSE,
    #                                                     scope='resnet_v1_152')

    # saver = tf.train.Saver()
    # saver.restore(tf.Session(), "/home/boy2/UCF101/src/resNet-152/vgg_19.ckpt")

    conv_1 = tf.layers.conv2d(inputs=input_layer, filters=1, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             kernel_size=[3, 1], padding="valid", activation=tf.nn.relu, name='conv1')

    flat = tf.reshape(conv_1, [-1, 2048])

    logits = tf.layers.dense(inputs=flat, units=num_unique_classes, name='logits')

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

    learning_rate = tf.train.exponential_decay(
        start_learning_rate,  # Base learning rate.
        tf.train.get_global_step() * batch_size,  # Current index into the dataset.
        params['train_set_size'] * epech_decay,  # Decay step.
        decay_rate,  # Decay rate.
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
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def classify(train_data, train_labels, eval_data, eval_labels):
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100000)

    params = {"train_set_size": len(train_data), "feature_len": len(train_data[0])}
    mnist_classifier = tf.estimator.Estimator(model_fn=model, params=params)

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

def norm_encode_data(train_data_label_image, test_data_label_image, encoder):
    len_train = len(train_data_label_image['data'])
    # normalize the data in time direction
    temp = np.concatenate([train_data_label_image['data'], test_data_label_image['data']])
    t, x, y, c = temp.shape
    temp = temp.reshape((t, x*y, c))
    temp_norm = np.zeros(temp.shape)
    for i in range(c):
        temp_norm[:, :, i] = preprocessing.normalize(temp[:, :, i], axis=0)
    temp_norm = temp_norm.reshape((t,x,y,c))
    # optional: normalize each sample data independently
    # temp_norm = preprocessing.normalize(temp_norm, axis=1)
    train_data_label_image['data'] = temp_norm[:len_train]
    test_data_label_image['data'] = temp_norm[len_train:]
    train_data_label_image['label'] = encoder.fit_transform(train_data_label_image['label'])
    test_data_label_image['label'] = encoder.fit_transform(test_data_label_image['label'])
    return train_data_label_image, test_data_label_image

def reformat(data_label_image, data_label_flow_1, data_label_flow_2):
    data = []
    for i, f1, f2 in zip(data_label_image['data'], data_label_flow_1['data'], data_label_flow_2['data']):
        temp = []
        # for j in range(len(i[0])):
        temp.append(i)
        temp.append(f1)
        temp.append(f2)
        data.append(temp)
    return {'data': np.array(data), 'label': data_label_image['label']}

def main(resNet_save_path, resNet_ws_save_path, ucf_resNet_flow_save_path_1, ucf_resNet_flow_ws_save_path_1,
         ucf_resNet_flow_save_path_2, ucf_resNet_flow_ws_save_path_2, train_test_splits_save_path, dim, dataset='ucf'):
    features_image = []
    features_flow_u = []
    features_flow_v = []
    for (dirpath, dirnames, filenames) in os.walk(resNet_ws_save_path):
        features_image += [f for f in filenames if f.endswith('.npy')]
    for (dirpath, dirnames, filenames) in os.walk(ucf_resNet_flow_ws_save_path_1):
        features_flow_u += [f for f in filenames if f.endswith('.npy')]
    for (dirpath, dirnames, filenames) in os.walk(ucf_resNet_flow_ws_save_path_2):
        features_flow_v += [f for f in filenames if f.endswith('.npy')]
    if len(features_image) == 0:
        calWeightedSum.calculate_weightedsum(resNet_save_path, resNet_ws_save_path, dim)
    if len(features_flow_u) == 0:
        calWeightedSum.calculate_weightedsum(ucf_resNet_flow_save_path_1, ucf_resNet_flow_ws_save_path_1, dim)
    if len(features_flow_v) == 0:
        calWeightedSum.calculate_weightedsum(ucf_resNet_flow_save_path_2, ucf_resNet_flow_ws_save_path_2, dim)

    # if len(features_flow_u) == 0 or len(features_flow_v) == 0:
    #     ws_flows(ucf_resNet_flow_save_path_1, ucf_resNet_flow_save_path_2, ucf_resNet_flow_ws_save_path_1,
    #              ucf_resNet_flow_ws_save_path_2, dim)

    if dataset == 'hmdb':
        tts = TrainTestSampleGen(ucf_path='', hmdb_path=train_test_splits_save_path)
    else:
        tts = TrainTestSampleGen(ucf_path=train_test_splits_save_path, hmdb_path='')

    acc = 0
    encoder = preprocessing.LabelEncoder()
    for i in range(1):
        # resNet image feature
        train_data_label_image, test_data_label_image = tts.train_test_split(resNet_ws_save_path, dataset, i, crop=True)
        # normalize the data and encode labels
        train_data_label_image, test_data_label_image = norm_encode_data(train_data_label_image, test_data_label_image,
                                                                         encoder)

        # for i in range(len(train_data_label_image['data'])):
        #     np.save('/home/boy2/UCF101/ucf101_dataset/features/temp/train/' + str(i) + '_' + str(train_data_label_image['label'][i]) + '.npy',
        #             train_data_label_image['data'][i])
        #
        # for i in range(len(test_data_label_image['data'])):
        #     np.save('/home/boy2/UCF101/ucf101_dataset/features/temp/test/' + str(i) + '_' + str(test_data_label_image['label'][i]) + '.npy',
        #             test_data_label_image['data'][i])

        # np.save('/home/boy2/UCF101/ucf101_dataset/features/temp/train.npy', train_data_label_image)
        # np.save('/home/boy2/UCF101/ucf101_dataset/features/temp/test.npy', test_data_label_image)

        train_data_label_image['data'], test_data_label_image['data'] = extract(train_data_label_image['data'],
                                                                                test_data_label_image['data'])

        # resNet flow u feature
        train_data_label_flow_1, test_data_label_flow_1 = tts.train_test_split(ucf_resNet_flow_ws_save_path_1, dataset,
                                                                               i, crop=True)
        # normalize the data and encode labels
        train_data_label_flow_1, test_data_label_flow_1 = norm_encode_data(train_data_label_flow_1,
                                                                           test_data_label_flow_1, encoder)
        train_data_label_flow_1['data'], test_data_label_flow_1['data'] = extract(train_data_label_flow_1['data'],
                                                                                  test_data_label_flow_1['data'])

        # resNet flow v feature
        train_data_label_flow_2, test_data_label_flow_2 = tts.train_test_split(ucf_resNet_flow_ws_save_path_2, dataset,
                                                                               i, crop=True)
        # normalize the data in time direction
        train_data_label_flow_2, test_data_label_flow_2 = norm_encode_data(train_data_label_flow_2,
                                                                           test_data_label_flow_2, encoder)
        train_data_label_flow_2['data'], test_data_label_flow_2['data'] = extract(train_data_label_flow_2['data'],
                                                                                  test_data_label_flow_2['data'])

        # combine the image and flow features together with different channel
        train_data_label = reformat(train_data_label_image, train_data_label_flow_1, train_data_label_flow_2)
        test_data_label = reformat(test_data_label_image, test_data_label_flow_1, test_data_label_flow_2)

        # train_data_label = reformat_flow(train_data_label_flow_1, train_data_label_flow_2)
        # test_data_label = reformat_flow(test_data_label_flow_1, test_data_label_flow_2)

        # train_data_label, test_data_label = norm_encode_data(train_data_label, test_data_label, encoder)

        # swape the axis
        # train_data_label['data'] = np.swapaxes(train_data_label['data'], 1, 3)
        # test_data_label['data'] = np.swapaxes(test_data_label['data'], 1, 3)

        # # resize the train and test data
        # trainx, trainy, trainz = train_data_label['data'].shape
        # train_data_label['data'] = np.reshape(train_data_label['data'], newshape=(trainx, trainy, trainz, 1))
        # testx, testy, testz = test_data_label['data'].shape
        # test_data_label['data'] = np.reshape(test_data_label['data'], newshape=(testx, testy, testz, 1))

        # train_data_label = {'data': np.concatenate(
        #     (train_data_label_image['data'], train_data_label_flow_1['data'], train_data_label_flow_2['data']), axis=1),
        #                     'label': train_data_label_image['label']}
        # test_data_label = {'data': np.concatenate(
        #     (test_data_label_image['data'], test_data_label_flow_2['data'], test_data_label_flow_2['data']), axis=1),
        #                    'label': test_data_label_image['label']}

        # train_data_label = {
        #     'data': train_data_label_image['data']+train_data_label_flow_1['data']+train_data_label_flow_2['data'],
        #     'label': train_data_label_image['label']}
        # test_data_label = {
        #     'data': test_data_label_image['data']+test_data_label_flow_1['data']+test_data_label_flow_2['data'],
        #     'label': test_data_label_image['label']}

        # acc += nn_classifier.classify(train_data_label['data'], train_data_label['label'],
        #                               test_data_label['data'], test_data_label['label'])['accuracy']
        # acc += twodconv_classifier.classify(train_data_label['data'], train_data_label['label'],
        #                                     test_data_label['data'], test_data_label['label'])['accuracy']
        acc += classify(train_data_label['data'], train_data_label['label'],
                        test_data_label['data'], test_data_label['label'])['accuracy']
        # acc += onedconv_classifier.classify(train_data_label_image['data'], train_data_label_image['label'],
        #                                     test_data_label_image['data'], test_data_label_image['label'])['accuracy']
    print('accuracy for ', dataset, 'is', acc / (i + 1))


if __name__ == '__main__':
    ucf_resNet_ws_save_path = "/home/boy2/UCF101/ucf101_dataset/features/frame_ws_3"
    ucf_resNet_flow_ws_save_path_1 = "/home/boy2/UCF101/ucf101_dataset/features/flow_ws_u_3"
    ucf_resNet_flow_ws_save_path_2 = "/home/boy2/UCF101/ucf101_dataset/features/flow_ws_v_3"
    ucf_resNet_flow_save_path_1 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow_crop/u"
    ucf_resNet_flow_save_path_2 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow_crop/v"
    ucf_resNet_save_path = "/home/boy2/UCF101/ucf101_dataset/features/resNet_crop"
    ucf_train_test_splits_save_path = "/home/boy2/UCF101/ucf101_dataset/features/testTrainSplits"
    main(ucf_resNet_save_path, ucf_resNet_ws_save_path, ucf_resNet_flow_save_path_1, ucf_resNet_flow_ws_save_path_1,
         ucf_resNet_flow_save_path_2, ucf_resNet_flow_ws_save_path_2, ucf_train_test_splits_save_path, 2)


