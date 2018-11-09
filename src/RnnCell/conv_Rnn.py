import tensorflow as tf


class ConvRnnCell(object):
    def __init__(self, num_units, feature_len, channel, mode):
        self._num_units = num_units
        self._feature_len = feature_len
        self._channel = channel
        self._mode = mode

    @property
    def state_size(self):
        return self._num_units*self._channel

    @property
    def output_size(self):
        return self._num_units*self._channel

    def zero_state(self, batch_size, dtype):
        return tf.zeros([batch_size, self._num_units*self._channel], dtype=dtype)

    def __call__(self, inputs, state, scope=None):
        inputs = tf.reshape(inputs, shape=(tf.shape(inputs)[0], 2048, self._channel))
        inputs = tf.expand_dims(inputs, 1)
        state = tf.reshape(state, shape=(tf.shape(state)[0], self._num_units, self._channel))
        state = tf.expand_dims(state, 1)
        # state = tf.expand_dims(state, 3)
        input_layer = tf.concat([inputs, state], axis=1)
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=16,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 use_bias=True, kernel_size=(1, 1), padding="valid", activation=tf.nn.selu,
                                 name='conv1')
        bn1 = tf.layers.batch_normalization(conv1, training=(self._mode == tf.estimator.ModeKeys.TRAIN))
        conv2 = tf.layers.conv2d(inputs=bn1, filters=8,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 use_bias=True, kernel_size=(2, 1), padding="valid", activation=tf.nn.selu,
                                 name='conv2')
        bn2 = tf.layers.batch_normalization(conv2, training=(self._mode == tf.estimator.ModeKeys.TRAIN))
        conv3 = tf.layers.conv2d(inputs=bn2, filters=3, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 use_bias=True, kernel_size=(1, 1), padding="valid", activation=None, name='conv3')
        bn3 = tf.layers.batch_normalization(conv3, training=(self._mode == tf.estimator.ModeKeys.TRAIN))
        outputs = tf.squeeze(bn3, 1)
        outputs = tf.reshape(outputs, shape=(tf.shape(inputs)[0], 2048*self._channel))
        return outputs, outputs
