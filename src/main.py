import os

import numpy as np
import tensorflow as tf
from sklearn import preprocessing

from No_use.frameFeatureCalculator import Framefeaturecalculator
from No_use.stipExtractor import Stipextractor
from No_use.ws_over_TD import WsOverTD
from frameLoader import Frameloader
from npyFileReader import Npyfilereader
from weighted_sum.videoDescriptorWeightedSum import Weightedsum

tf.logging.set_verbosity(tf.logging.INFO)

start_learning_rate = 0.001
batch_size = 4096*2
dropout_rate = 0.8
alpha = 0.2
num_unique_classes = 101
test_data_ratio = 0.3
epech_decay = 3
decay_rate = 0.9
epech = 50


def remove_problem_file(f_path, d_path):
    with open(f_path) as f:
        contents = f.readlines()
    for line in contents:
        p = os.path.join(d_path, line.split(".")[0] + '.npy')
        if os.path.exists(p):
            os.remove(p)


# Done! partial
def cal_save_stip_feature(frame_path, stips_path, save_path):
    ffc = Framefeaturecalculator(frame_path, stips_path, save_path)
    ffc.validate()
    video_num = len(ffc.frame_loader.frame_parent_paths)
    for i in range(video_num):
        if ffc.cal_save_features() == None:
            print(ffc.current_video_name)


# Done! partial
def calculate_surf_weightedsum(s_path, W_path):
    nr = Npyfilereader(s_path)
    nr.validate(W_path)
    video_num = len(nr.npy_paths)
    for i in range(video_num):
        name, contents = nr.read_npys()
        ws = Weightedsum(name, contents, W_path)
        # if ws.pre_processing() == -1:
        #     print(name)
        #     problem_path.append(name)
        #     continue
        ws.ws_descriptor_gen(5)


def read_surfWS_tar(path):
    surfWS_path = []
    tar = []
    surfWS_des = []
    fn = []
    fs_dict = {}
    for (dirpath, dirnames, filenames) in os.walk(path):
        surfWS_path += [os.path.join(dirpath, f) for f in filenames if f != '.DS_Store']
        tar += [f.split()[0].split('_')[0] for f in filenames if f != '.DS_Store']
        fn += [f for f in filenames if f != '.DS_Store']
    for surfWS in surfWS_path:
        surfWS_des.append(np.load(surfWS))
    surfWS_des = preprocessing.normalize(np.array(surfWS_des), norm='l2')
    for f, s in zip(fn, surfWS_des):
        fs_dict[f.split('.')[0]] = s
    return fs_dict, tar, surfWS_des


def read_surfWS_tar_1(path):
    surfWS_path = []
    tar = []
    surfWS_des = []
    fn = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        surfWS_path += [os.path.join(dirpath, f) for f in filenames if f != '.DS_Store']
        tar += [f.split()[0].split('_')[0] for f in filenames if f != '.DS_Store']
        fn += [f for f in filenames if f != '.DS_Store']
    for f, surfWS in zip(fn, surfWS_path):
        surfWS_des.append(np.load(surfWS))
    return np.array(surfWS_des), np.array(tar)


def read_feature_label_by_list(file_path, data_store_path, mode):
    feature = []
    label = []
    with open(file_path) as f:
        data_names_labels = f.readlines()
    for ele in data_names_labels:
        if mode == 'train':
            n, _ = ele.split(" ")
            label.append(n.split("/")[0])
            data_feature_file = n.split("/")[0] + '_' + n.split("/")[-1].split('.')[0] + '.npy'
            feature.append(np.load(os.path.join(data_store_path, data_feature_file)))
        else:
            l = ele.split('/')[0]
            label.append(l)
            data_feature_file = l + '_' + ele.split("/")[-1].split('.')[0] + '.npy'
            feature.append(np.load(os.path.join(data_store_path, data_feature_file)))
    return np.array(feature), np.array(label)


def find_train_test(des, train_file, mode):
    feature = []
    label = []
    with open(train_file) as f:
        data_names_labels = f.readlines()
    for ele in data_names_labels:
        if mode == 'train':
            n, _ = ele.split(" ")
            l, n = n.split("/")
            label.append(l)
            feature.append(des[l + '_' + n.split('.')[0]])
        else:
            l, n = ele.split("/")
            label.append(l)
            feature.append(des[l + '_' + n.split('.')[0]])
    return np.array(feature), np.array(label)


def find_stip(frame_path, save_path, sigma, tau, scale, k):
    frame_loader = Frameloader(frame_path)
    for p in frame_loader.frame_parent_paths:
        video_name = p.split('/')[-1]
        stip_path = os.path.join(save_path,
                                 video_name + '.npy')
        if os.path.exists(stip_path):
            continue
        frames = frame_loader.load_frames(p)
        stips = Stipextractor(frames, sigma=sigma, tau=tau, scale=scale, k=k).laptev_stip_extractor()
        np.save(stip_path, stips)
        # with open(stip_path, 'w') as file:
        #     for i in range(len(stips)):
        #         co = str(stips[i][0]) + ',' + str(stips[i][1]) + ',' + str(stips[i][2])
        #         file.write(co)
        #         if i != len(stips) - 1:
        #             file.write('\n')


def transformation_matrix_gen(r, c):
    """
    Generate the transformation matrix. The transformation matrix has dimension r*c and r<=c. Also, the transformation
    matrix contain r linear independent column vectors.
    :param r: the row number
    :param c: the column number
    :return: the transformation matrix
    """
    iden = np.identity(r)
    random_matrix = np.random.randint(10, size=(r, c - r))
    return np.concatenate((iden, np.matmul(iden, random_matrix)), axis=1)


def cal_ws_over_TD(video_path, num_weight_vectors, ws_store_path):
    fl = Frameloader(video_path)
    # fl.validate(ws_store_path)
    for i in range(len(fl.frame_parent_paths)):
        video_name = fl.get_current_video_name()
        frames = fl.load_frames()
        weights = transformation_matrix_gen(num_weight_vectors, len(frames))
        ws = WsOverTD(np.array(frames), weights).sum_with_weights()
        if ws is not None:
            np.save(os.path.join(ws_store_path, video_name + '.npy'), ws)
        else:
            print(video_name)
        if i % 100 == 0 and i >= 100:
            print("Already processed " + str(i) + " videos!")


def cal_mean(s_path, W_path):
    nr = Npyfilereader(s_path)
    nr.validate(W_path)
    video_num = len(nr.npy_paths)
    for i in range(video_num):
        name, contents = nr.read_npys()
        ws = Weightedsum(name, contents, W_path)
        # if ws.pre_processing() == -1:
        #     print(name)
        #     problem_path.append(name)
        #     continue
        ws.mean_descriptor_gen()


def simple_model(features, labels, mode, params):
    input_layer = tf.reshape(features['x'], [-1, params["feature_len"]])
    input_layer = tf.cast(input_layer, tf.float32)
    # dense1 = tf.layers.dense(inputs=input_layer, units=8000, activation=tf.nn.relu)
    # dropout1 = tf.layers.dropout(
    #     inputs=dense1, rate=0.8, training=mode == tf.estimator.ModeKeys.TRAIN)
    #
    # Dense Layer 2
    fc0 = tf.layers.dense(inputs=input_layer, units=5000, activation=None)
    relu0 = tf.nn.leaky_relu(fc0, alpha=alpha)
    dropout0 = tf.layers.dropout(
        inputs=relu0, rate=dropout_rate, training=mode == tf.estimator.ModeKeys.TRAIN)
    #
    # Dense Layer 3
    fc1 = tf.layers.dense(inputs=relu0, units=3000, activation=None)
    bn1 = tf.layers.batch_normalization(fc1, axis=1, center=True, scale=True,
                                        training=mode == tf.estimator.ModeKeys.TRAIN)
    relu1 = tf.nn.leaky_relu(fc1, alpha=alpha)
    # dropout0 = tf.layers.dropout(
    #     inputs=bn0, rate=0.8, training=mode == tf.estimator.ModeKeys.TRAIN)
    #
    fc2 = tf.layers.dense(inputs=relu1, units=1250, activation=None)
    bn2 = tf.layers.batch_normalization(fc2, axis=1, center=True, scale=True,
                                        training=mode == tf.estimator.ModeKeys.TRAIN)
    relu2 = tf.nn.leaky_relu(fc2, alpha=alpha)
    # dropout1 = tf.layers.dropout(
    #     inputs=relu1, rate=0.8, training=mode == tf.estimator.ModeKeys.TRAIN)

    fc3 = tf.layers.dense(inputs=relu2, units=625, activation=None)
    bn3 = tf.layers.batch_normalization(fc3, axis=1, center=True, scale=True,
                                        training=(mode == tf.estimator.ModeKeys.TRAIN))
    relu3 = tf.nn.leaky_relu(fc3, alpha=alpha)
    dropout3 = tf.layers.dropout(
        inputs=relu3, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout3, units=num_unique_classes)
    # Network construction done!

    # if mode == tf.estimator.ModeKeys.PREDICT:
    #     return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

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
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def create_batches(data, labels, batch_size):
    num_chunks = np.ceil(len(data) / batch_size)
    d = np.array_split(np.array(data), num_chunks)
    l = np.array_split(np.array(labels), num_chunks)
    return d, l


def shuffle(data, labels):
    """
    Randomly shuffle the elements in paths_list
    :param paths_list: path list
    :return: shuffled list
    """
    idx = np.random.randint(0, len(data), len(data))
    d = np.array(data)[idx]
    l = np.array(labels)[idx]
    return d, l


if __name__ == '__main__':
    f_path = "/home/boy2/UCF101/UCF_101_dataset/UCF101_frames"
    s_path = "/home/boy2/UCF101/laptev_stips_25fps"
    # save_path = "/home/boy2/UCF101/laptev_stips_25fps_sruf"
    save_path = "/home/boy2/UCF101/VGGNet_frame_features"
    # save_path = "/home/boy2/UCF101/VGGNet_hmdb51_frame_features"

    # WS_path = "/home/boy2/UCF101/laptev_stips_25fps_srufWS"
    WS_path_mean = "/home/boy2/UCF101/VGGNet_mean"
    WS_path_WS = "/home/boy2/UCF101/VGGNet_WS"
    # WS_path_WS = '/home/boy2/UCF101/VGGNet_hmdb51_WS'
    # WS_path_mean = '/home/boy2/UCF101/VGGNet_hmdb51_mean'
    train_data_file = ['/home/boy2/UCF101/UCF_101_dataset/ucfTrainTestlist/trainlist01.txt',
                       '/home/boy2/UCF101/UCF_101_dataset/ucfTrainTestlist/trainlist02.txt',
                       '/home/boy2/UCF101/UCF_101_dataset/ucfTrainTestlist/trainlist03.txt']
    test_data_file = ['/home/boy2/UCF101/UCF_101_dataset/ucfTrainTestlist/testlist01.txt',
                      '/home/boy2/UCF101/UCF_101_dataset/ucfTrainTestlist/testlist02.txt',
                      '/home/boy2/UCF101/UCF_101_dataset/ucfTrainTestlist/testlist03.txt']

    calculate_surf_weightedsum(save_path, WS_path_WS)
    cal_mean(save_path, WS_path_mean)
    des, tar, _ = read_surfWS_tar(WS_path_WS)
    tar = preprocessing.LabelEncoder().fit_transform(np.array(tar))
    encoder = preprocessing.LabelEncoder()
    for train, test in zip(train_data_file, test_data_file):
        # 9537, 9586, 9624
        # train_data, train_labels = read_feature_label_by_list(train, WS_path_WS, 'train')
        # eval_data, eval_labels = read_feature_label_by_list(test, WS_path_WS, 'test')
        # norm = preprocessing.normalize(np.concatenate([train_data, eval_data], axis=0), norm='l2', axis=0)
        # train_data = norm[: len(train_data)]
        # eval_data = norm[len(train_data):]
        # train_labels = encoder.fit_transform(train_labels)
        # eval_labels = encoder.fit_transform(eval_labels)

        # # # des, tar = shuffle(des, tar)
        # des = preprocessing.normalize(np.array(des), norm='l2')
        train_data, train_labels = find_train_test(des, train, 'train')
        eval_data, eval_labels = find_train_test(des, test, 'test')
        train_labels = encoder.fit_transform(train_labels)
        eval_labels = encoder.fit_transform(eval_labels)
        train_data, train_labels = shuffle(train_data, train_labels)
        eval_data, eval_labels = shuffle(eval_data, eval_labels)

        # # tar = preprocessing.OneHotEncoder(sparse=False).fit_transform(tar.reshape(len(tar), 1))
        #
        # print('classifiering')
        #
        # train_data, eval_data, train_labels, eval_labels = model_selection.train_test_split(des, tar,
        #                                                                                     test_size=test_data_ratio,
        #                                                                                     random_state=42)
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=1000)
        params = {"train_set_size": len(train_data), "feature_len": len(train_data[0])}
        mnist_classifier = tf.estimator.Estimator(model_fn=simple_model, params=params)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=batch_size,
            num_epochs=epech,
            shuffle=True)
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=None,
            hooks=[logging_hook])

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            batch_size=len(eval_data),
            num_epochs=1,
            shuffle=False)
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)
