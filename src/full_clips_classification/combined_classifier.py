import tensorflow as tf
import numpy as np
from models.research.slim.nets import resnet_v1

slim = tf.contrib.slim


def build_spatial(x, reuse, temporal_length, mode, batch_size):
    with tf.device('/gpu:1'):
        # data argumentation * 10
        shape = x.get_shape().as_list()
        x = tf.reshape(x, [batch_size * shape[1], shape[2], shape[3], shape[4]])

        # cropped_image = list()
        # for i in range(shape[0]):
        #     for b in boxes[i]:
        #         # original frame
        #         cropped_image.append(tf.image.crop_and_resize(x[i], boxes=[b for _ in range(temporal_length)],
        #                              box_ind=[j for j in range(temporal_length)], crop_size=[224, 224]))
        #         # flipped frame
        #         cropped_image.append(tf.image.crop_and_resize(tf.image.flip_left_right(x[i]), boxes=[b for _ in range(temporal_length)],
        #                                                       box_ind=[j for j in range(temporal_length)],
        #                                                       crop_size=[224, 224]))
        # x = tf.stack(cropped_image)
        # shape = x.get_shape().as_list()
        # x = tf.reshape(x, [shape[0]*shape[1], shape[2], shape[3], shape[4]])

        with tf.variable_scope('ConvNet_s', reuse=reuse):
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                resNet152, end_points = resnet_v1.resnet_v1_50(x,
                                                               num_classes=None,
                                                               is_training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                               global_pool=True,
                                                               output_stride=None,
                                                               spatial_squeeze=True,
                                                               reuse=None,
                                                               scope='resnet_v1_152')

            shape = resNet152.get_shape().as_list()
            input_rgb = tf.reshape(resNet152, [int(shape[0] / temporal_length), temporal_length, 2048, 1])

            conv1 = tf.layers.conv2d(inputs=input_rgb, filters=16,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     use_bias=True, kernel_size=[temporal_length, 1], padding="same", activation=None,
                                     strides=[temporal_length, 1],
                                     name='conv1_s')
            conv_bn1 = tf.layers.batch_normalization(conv1, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                     name='conv_bn1_s')
            conv_act1 = tf.nn.relu(conv_bn1, name="conv_act1_s")

            conv_act1 = tf.transpose(conv_act1, [0, 3, 2, 1])

            conv2 = tf.layers.conv2d(inputs=conv_act1, filters=8,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     use_bias=True, kernel_size=[1, 1], padding="same",
                                     activation=None, strides=[1, 1],
                                     name='conv2_s')
            conv_bn2 = tf.layers.batch_normalization(conv2, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                     name='conv_bn2_s')
            conv_act2 = tf.nn.relu(conv_bn2, name="conv_act2_s")

            conv3 = tf.layers.conv2d(
                inputs=conv_act2,
                filters=4,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                use_bias=False, kernel_size=[16, 1], padding="same", activation=None, strides=[16, 1],
                name='conv3_s')
            conv_bn3 = tf.layers.batch_normalization(conv3, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                     name='conv_bn3_s')
            conv_act3 = tf.nn.relu(conv_bn3, name="conv_act3_s")

            return conv_act3


def build_temporal(x, reuse, temporal_length, mode, batch_size):
    with tf.device('/gpu:1'):
        shape = x.get_shape().as_list()
        x = tf.reshape(x, [batch_size * shape[1], shape[2], shape[3], shape[4]])

        # cropped_image = list()
        # for i in range(shape[0]):
        #     for b in boxes[i]:
        #         # original frame
        #         original = tf.reshape(x[i], [shape[1]*shape[2], shape[3], shape[4], shape[5]])
        #         cropped_image.append(tf.image.crop_and_resize(original, boxes=[b for _ in range(temporal_length*shape[2])],
        #                                                       box_ind=[j for j in range(temporal_length*shape[2])],
        #                                                       crop_size=[224, 224]))
        #         # flipped frame
        #         flipped = tf.image.flip_left_right(original)
        #         cropped_image.append(
        #             tf.image.crop_and_resize(flipped, boxes=[b for _ in range(temporal_length*shape[2])],
        #                                      box_ind=[j for j in range(temporal_length*shape[2])],
        #                                      crop_size=[224, 224]))
        # x = tf.stack(cropped_image)
        # shape = x.get_shape().as_list()
        # x = tf.reshape(x, [shape[0] * shape[1], shape[2], shape[3], shape[4]])

        with tf.variable_scope('ConvNet_t', reuse=reuse):
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                resNet152, end_points = resnet_v1.resnet_v1_50(x,
                                                               num_classes=None,
                                                               is_training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                               global_pool=True,
                                                               output_stride=None,
                                                               spatial_squeeze=True,
                                                               reuse=None,
                                                               scope='resnet_v1_152')

            shape = resNet152.get_shape().as_list()
            input_flow = tf.reshape(resNet152,
                                    [int(shape[0] / 20 / temporal_length), 20 * temporal_length, 2048, 1])

            # input_flow = tf.transpose(input_flow, [0, 1, 3, 2])
            # input_flow = tf.reshape(input_flow, [-1, params['flow_feature_shape'][2], 2048 * 2, 1])
            # input_flow = tf.reshape(input_flow, [-1, params['flow_feature_shape'][2]*2, 2048, 1])
            conv4 = tf.layers.conv2d(inputs=input_flow, filters=16,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     use_bias=False, kernel_size=[20, 1], padding="same", activation=None,
                                     strides=[20, 1],
                                     name='conv4')
            conv_bn4 = tf.layers.batch_normalization(conv4, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                     name='conv_bn4')
            conv_act4 = tf.nn.relu(conv_bn4, name="conv_act4")

            conv4_a = tf.layers.conv2d(inputs=conv_act4, filters=16,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       use_bias=False, kernel_size=[temporal_length, 1], padding="same",
                                       activation=None, strides=[temporal_length, 1],
                                       name='conv4_a_t')
            conv_bn4_a = tf.layers.batch_normalization(conv4_a, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                       name='conv_bn4_a')
            conv_act4_a = tf.nn.relu(conv_bn4_a, name="conv_act4_a_t")
            conv_act4_a = tf.transpose(conv_act4_a, [0, 3, 2, 1])

            conv5 = tf.layers.conv2d(
                inputs=conv_act4_a,
                filters=8,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                use_bias=False, kernel_size=[1, 1], padding="same", activation=None, strides=[1, 1],
                name='conv5_t')
            conv_bn5 = tf.layers.batch_normalization(conv5, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                     name='conv_bn5_t')
            conv_act5 = tf.nn.relu(conv_bn5, name="conv_act5_t")

            conv6 = tf.layers.conv2d(inputs=conv_act5, filters=4,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     use_bias=False, kernel_size=[16, 1], padding="same", activation=None,
                                     strides=[16, 1],
                                     name='conv6_t')
            conv_bn6 = tf.layers.batch_normalization(conv6, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                     name='conv_bn6_t')
            conv_act6 = tf.nn.relu(conv_bn6, name="conv_act6_t")
            return conv_act6


def combine_two_stream(spatial, temporal, reuse, mode, dropout_rate, num_unique_classes):
    with tf.device('/gpu:1'):
        with tf.variable_scope('Combine', reuse=reuse):
            res = tf.concat([spatial, temporal], axis=2)

            flat_rgb = tf.reshape(spatial, [-1, 1 * 2048 * 4], name='flat_rgb')
            with tf.device('/gpu:1'):
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

            logits = tf.layers.dense(inputs=dropoutfc2, units=num_unique_classes, use_bias=False, name='logits_rgb')
            return logits


def build_model(rgb_input, flow_input, temporal_length, reuse, mode, dropout_rate, num_unique_classes, batch_size):
    with tf.device('/gpu:1'):
        spatial = build_spatial(rgb_input, reuse, temporal_length, mode, batch_size)
        temporal = build_temporal(flow_input, reuse, temporal_length, mode, batch_size)
        model = combine_two_stream(spatial, temporal, reuse, mode, dropout_rate, num_unique_classes)
        return model
