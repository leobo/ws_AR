import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
from time import time
import tensorflow as tf
import concurrent.futures
import itertools
from sklearn import preprocessing, svm

from trainTestSamplesGen import TrainTestSampleGen
from weighted_sum.videoDescriptorWeightedSum import Weightedsum
from ws_unbalanced_video_clips.classifier_v3 import classify

# from ws_unbalanced_video_clips.two_stream_classifier import classify
train_steps = 10
test_steps = 5
dims = 8
num_cha = 9
min_len_video = 30
max_len_video = 1000
# split 1
num_train_data = 101*10
num_test_data = 3783
# # split 2
# num_train_data = 9586
# num_test_data = 3734
# # split 3
# num_train_data = 9624
# num_test_data = 3696
# # test
# num_train_data = 10
# num_test_data = 10
# num_train = np.random.randint(0, 3783, size=10)
# num_test = np.random.randint(0, 3783, size=10)
num_trans_matrices = 1
num_samples_per_training_video = 10
num_samples_per_testing_video = 10


def fisher_vector(xx, gmm):
    """Computes the Fisher vector on a set of descriptors.
    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors
    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.
    Returns
    -------
    fv: array_like, shape (K + 2 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.
    Reference
    ---------
    J. Krapac, J. Verbeek, F. Jurie.  Modeling Spatial Layout with Fisher
    Vectors for Image Categorization.  In ICCV, 2011.
    http://hal.inria.fr/docs/00/61/94/03/PDF/final.r1.pdf
    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
            - Q_xx_2
            - Q_sum * gmm.means_ ** 2
            + Q_sum * gmm.converged_
            + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))


def norm_encode_data(train_data, test_data):
    len_train = len(train_data)
    # normalize the data in time direction
    temp = np.concatenate([train_data, test_data])
    if len(temp.shape) == 3:
        x, y, t = temp.shape
        temp_norm = np.zeros(shape=(x, y, t))
        for i in range(t):
            temp_norm[:, :, i] = preprocessing.normalize(temp[:, :, i], axis=0, norm='max')
            # temp_norm[:, :, i] = preprocessing.normalize(temp_norm[:, :, i], axis=1, norm='max')
            # temp_norm[:, :, i] = preprocessing.scale(temp[:, :, i], axis=0)
    else:
        temp_norm = preprocessing.normalize(temp, axis=1, norm='max')
    # optional: normalize each sample data independently
    # temp_norm = preprocessing.normalize(temp_norm, axis=1)
    train_data = temp_norm[:len_train]
    test_data = temp_norm[len_train:]
    return train_data, test_data


def find_max(train_data, test_data):
    # normalize the data in time direction
    temp = np.concatenate([train_data, test_data])
    m = []
    if len(temp.shape) == 3:
        x, y, t = temp.shape
        for i in range(t):
            m.append(np.max(temp[:, :, i], axis=0))
    else:
        temp_norm = preprocessing.normalize(temp, axis=1, norm='max')
    return np.max(m, axis=0)
    # return m


def scaler_encode_data(train_data, test_data):
    len_train = len(train_data)
    # normalize the data in time direction
    temp = np.concatenate([train_data, test_data])
    if len(temp.shape) == 3:
        x, y, t = temp.shape
        temp_norm = np.zeros(shape=(x, y, t))
        scaler = preprocessing.StandardScaler()
        for i in range(t):
            scaler.fit(temp_norm[:, :, i])
        for i in range(t):
            temp_norm[:, :, i] = scaler.transform(temp[:, :, i])

    else:
        temp_norm = preprocessing.normalize(temp, axis=1, norm='max')
    # optional: normalize each sample data independently
    # temp_norm = preprocessing.normalize(temp_norm, axis=1)
    train_data = temp_norm[:len_train]
    test_data = temp_norm[len_train:]
    return train_data, test_data


def norm_encode_data_v2(train_data_1, test_data_1, train_data_2, test_data_2):
    len_train = len(train_data_1)
    len_test = len(test_data_1)
    # normalize the data in time direction
    temp = np.concatenate([train_data_1, train_data_2, test_data_1, test_data_2])
    if len(temp.shape) == 3:
        x, y, t = temp.shape
        temp_norm = np.zeros(shape=(x, y, t))
        for i in range(t):
            temp_norm[:, :, i] = preprocessing.normalize(temp[:, :, i], axis=0, norm='max')
            # temp_norm[:, :, i] = preprocessing.normalize(temp_norm[:, :, i], axis=1, norm='max')
            # temp_norm[:, :, i] = preprocessing.scale(temp[:, :, i], axis=0)
    else:
        temp_norm = preprocessing.normalize(temp, axis=1, norm='max')
    # optional: normalize each sample data independently
    # temp_norm = preprocessing.normalize(temp_norm, axis=1)
    train_data_1 = temp_norm[:len_train]
    train_data_2 = temp_norm[len_train:2 * len_train]
    test_data_1 = temp_norm[2 * len_train:2 * len_train + len_test]
    test_data_2 = temp_norm[2 * len_train + len_test:]
    return train_data_1, test_data_1, train_data_2, test_data_2


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def tf_record_writer_eval_v2(path, name, rgb1, flow1, label1, rgb2, flow2, label2):
    if not os.path.exists(path):
        os.mkdir(path)
    writer = tf.python_io.TFRecordWriter(os.path.join(path, name))
    for _rgb1, _flow1, _label1, _rgb2, _flow2, _label2 in zip(rgb1, flow1, label1, rgb2, flow2, label2):
        feature = {'rgb': _bytes_feature(_rgb1.astype(np.float32).tobytes()),
                   'flow': _bytes_feature(_flow1.astype(np.float32).tobytes()),
                   'labels': _int64_feature(_label1)
                   }
        # print(feature)
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

        feature = {'rgb': _bytes_feature(_rgb2.astype(np.float32).tobytes()),
                   'flow': _bytes_feature(_flow2.astype(np.float32).tobytes()),
                   'labels': _int64_feature(_label2)
                   }
        # print(feature)
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()


def tf_record_writer_ch(path, name, rgb, flow, label, dims, num_cha):
    if not os.path.exists(path):
        os.mkdir(path)
    writer = tf.python_io.TFRecordWriter(os.path.join(path, name))
    for _rgb, _flow, _label in zip(rgb, flow, label):
        for i in range(int(dims / num_cha)):
            feature = {'rgb': _bytes_feature(_rgb.astype(np.float32).tobytes()),
                       'flow': _bytes_feature(_flow.astype(np.float32).tobytes()),
                       'labels': _int64_feature(_label)
                       }
            # print(feature)
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
    writer.close()


def tf_record_writer(path, name, rgb, flow, label):
    if not os.path.exists(path):
        os.mkdir(path)
    writer = tf.python_io.TFRecordWriter(os.path.join(path, name))
    for _rgb, _flow, _label in zip(rgb, flow, label):
        feature = {'rgb': _bytes_feature(_rgb.astype(np.float32).tobytes()),
                   'flow': _bytes_feature(_flow.astype(np.float32).tobytes()),
                   'labels': _int64_feature(_label)
                   }
        # print(feature)
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()


def tf_record_writer_eval(path, name, rgb1, flow1, label1, rgb2, flow2, label2, rgb3, flow3, label3):
    if not os.path.exists(path):
        os.mkdir(path)
    writer = tf.python_io.TFRecordWriter(os.path.join(path, name))
    for _rgb1, _flow1, _label1, _rgb2, _flow2, _label2, _rgb3, _flow3, _label3 in zip(rgb1, flow1, label1, rgb2, flow2,
                                                                                      label2, rgb3, flow3, label3):
        feature = {'rgb': _bytes_feature(_rgb1.astype(np.float32).tobytes()),
                   'flow': _bytes_feature(_flow1.astype(np.float32).tobytes()),
                   'labels': _int64_feature(_label1)
                   }
        # print(feature)
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

        feature = {'rgb': _bytes_feature(_rgb2.astype(np.float32).tobytes()),
                   'flow': _bytes_feature(_flow2.astype(np.float32).tobytes()),
                   'labels': _int64_feature(_label2)
                   }
        # print(feature)
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

        feature = {'rgb': _bytes_feature(_rgb3.astype(np.float32).tobytes()),
                   'flow': _bytes_feature(_flow3.astype(np.float32).tobytes()),
                   'labels': _int64_feature(_label3)
                   }
        # print(feature)
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()


def feature_gen(rp, name, seed, dim):
    rgb = np.load(rp)

    if len(seed.shape) == 1:
        tran_m_rgb = Weightedsum('tran_m', [], None).transformation_matrix_gen_norm(dims, len(rgb), seed)
        tran_m_rgb /= len(rgb)
        tran_m_rgb = preprocessing.normalize(tran_m_rgb, axis=1, norm='l2')
    else:
        tran_m_rgb = seed

    _rgb = Weightedsum(name, rgb[:], None).ws_descriptor_gen(dim, False, tran_m_rgb)
    return _rgb


def input_gen_all(rgb_feature_path, u_feature_path, v_feature_path, video_names, video_labels,
                  seed, steps=5, dim=2, train=False):
    video_names = np.array(video_names)
    video_labels = video_labels[:]
    video_names = video_names[:]

    # 1776 the max len video
    rgb_path = [os.path.join(rgb_feature_path, n + '.npy') for n in video_names]
    u_path = [os.path.join(u_feature_path, n + '.npy') for n in video_names]
    v_path = [os.path.join(v_feature_path, n + '.npy') for n in video_names]

    # with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    #     all_rgb = list(executor.map(feature_gen, rgb_path, video_names, itertools.repeat(seed, len(rgb_path)), itertools.repeat(dim, len(rgb_path))))
    #     all_u = list(executor.map(feature_gen, u_path, video_names, itertools.repeat(seed, len(u_path)), itertools.repeat(dim, len(u_path))))
    #     all_v = list(executor.map(feature_gen, v_path, video_names, itertools.repeat(seed, len(v_path)), itertools.repeat(dim, len(v_path))))

    all_rgb, all_u, all_v = list(), list(), list()
    for rp, up, vp, name, l in zip(rgb_path, u_path, v_path, video_names,
                                       video_labels):
        rgb = np.load(rp)
        u = np.load(up)
        v = np.load(vp)

        if len(seed.shape) == 1:
            tran_m_rgb = Weightedsum('tran_m', [], None).transformation_matrix_gen_norm(dims, len(rgb), seed)
            tran_m_rgb /= len(rgb)
            tran_m_rgb = preprocessing.normalize(tran_m_rgb, axis=1, norm='l2')
            # tran_m_flow = Weightedsum('tran_m', [], None).transformation_matrix_gen_norm(dims, len(u), seed)
            # tran_m_flow = preprocessing.normalize(tran_m_flow, axis=1, norm='l2')
        else:
            tran_m_rgb = seed
            # tran_m_flow = seed

        # m = np.min(rgb)
        # rgb += -m
        # # rgb = rgb / np.max(rgb)
        # m = np.min(u)
        # u += -m
        # # u = u / np.max(u)
        # m = np.min(v)
        # v += -m
        # # v = v / np.max(v)

        # select_frame = np.load(sf)
        # if select_frame.size != 0:
        #     select_frame[0].sort()
        #     if select_frame[0][-1] == len(rgb) + 1:
        #         select_frame[0] = select_frame[0][:-1]
        #     rgb = rgb[np.array(select_frame[1]) - 1]
        #     u = u[np.array(select_frame[1]) - 1]
        #     v = v[np.array(select_frame[2]) - 1]

        # while len(u) < tran_m.shape[-1]:
        #     rgb = np.concatenate((rgb, rgb))
        #     u = np.concatenate((u, u))
        #     v = np.concatenate((v, v))
        # rgb = rgb[:tran_m.shape[-1]]
        # u = u[:tran_m.shape[-1]]
        # v = v[:tran_m.shape[-1]]

        # # Time Varying Mean Vectors
        # copy_rgb = rgb.copy()
        # copy_u = u.copy()
        # copy_v = v.copy()
        # for r in range(len(rgb)):
        #     rgb[r] = np.sum(copy_rgb[:r + 1, :], axis=0) / (r + 1)
        #     rgb[r] = rgb[r] / np.linalg.norm(rgb[r])
        #     u[r] = np.sum(copy_u[:r + 1, :], axis=0) / (r + 1)
        #     u[r] = u[r] / np.linalg.norm(u[r])
        #     v[r] = np.sum(copy_v[:r + 1, :], axis=0) / (r + 1)
        #     v[r] = v[r] / np.linalg.norm(v[r])

        # tran_m_rgb = Weightedsum(None, None, None).attension_weights_gen(dims, len(rgb), seed)
        # tran_m_flow = Weightedsum(None, None, None).attension_weights_gen(dims, len(u), seed)

        _rgb = Weightedsum(name, rgb[:], None).ws_descriptor_gen(dim, False, tran_m_rgb)
        _u = Weightedsum(name, u[:], None).ws_descriptor_gen(dim, False, tran_m_rgb)
        _v = Weightedsum(name, v[:], None).ws_descriptor_gen(dim, False, tran_m_rgb)

        # # find x for Ax=0
        # _rgb = np.linalg.solve(np.transpose(_rgb), np.zeros(len(_rgb)))
        # _u = np.linalg.solve(np.transpose(_u), np.zeros(len(_u)))
        # _v = np.linalg.solve(np.transpose(_v), np.zeros(len(_v)))

        # # length of mapping on subspace defined by tran_m
        # _rgb = np.matmul(_rgb, tran_m)
        # _u = np.matmul(_u, tran_m)
        # _v = np.matmul(_v, tran_m)
        #
        # _rgb = Weightedsum(name, np.transpose(_rgb)[:], None).ws_descriptor_gen(dim, False, tran_m1)
        # _u = Weightedsum(name, np.transpose(_u)[:], None).ws_descriptor_gen(dim, False, tran_m1)
        # _v = Weightedsum(name, np.transpose(_v), None).ws_descriptor_gen(dim, False, tran_m1)

        all_rgb.append(_rgb)
        all_u.append(_u)
        all_v.append(_v)

    return all_rgb, all_u, all_v, video_labels


def input_gen_mul(rgb_feature_path, u_feature_path, v_feature_path, video_names, video_labels, steps=5, dim=2,
                  train=False):
    # 1775 the max len video
    rgb_path = [os.path.join(rgb_feature_path, n + '.npy') for n in video_names][:]
    u_path = [os.path.join(u_feature_path, n + '.npy') for n in video_names][:]
    all_rgb = []
    all_u = []
    all_label = []
    tran_m = Weightedsum('tran_m', [], None).transformation_matrix_gen(dim, 1776)
    tran_m = preprocessing.normalize(tran_m, axis=1)
    # tran_m = None
    for rp, up, name, l in zip(rgb_path, u_path, video_names[:], video_labels[:]):
        if rp != "/home/boy2/ucf101/ucf101_dataset/features/mulNet_feature/mulNet/rgb/o/v_StillRings_g21_c05.npy":
            rgb = np.load(rp)
            rgb = np.transpose(rgb)
            u = np.load(up)
            u = np.transpose(u)
            if len(rgb) > len(u) and len(rgb) - 1 == len(u):
                rgb = rgb[:-1]

            # # Time Varying Mean Vectors
            # copy_rgb = rgb.copy()
            # copy_u = u.copy()
            # copy_v = v.copy()
            # for r in range(len(rgb)):
            #     rgb[r] = np.sum(copy_rgb[:r + 1, :], axis=0) / (r + 1)
            #     rgb[r] = rgb[r] / np.linalg.norm(rgb[r])
            #     u[r] = np.sum(copy_u[:r + 1, :], axis=0) / (r + 1)
            #     u[r] = u[r] / np.linalg.norm(u[r])
            #     v[r] = np.sum(copy_v[:r + 1, :], axis=0) / (r + 1)
            #     v[r] = v[r] / np.linalg.norm(v[r])

            _rgb = Weightedsum(name, rgb[:], None).ws_descriptor_gen(dim, False, tran_m)
            _u = Weightedsum(name, u[:], None).ws_descriptor_gen(dim, False, tran_m)

            all_rgb.append(_rgb / len(rgb))
            all_u.append(_u / len(u))
            all_label.append(l)
    return all_rgb, all_u, all_label


def prdict(train_data, train_label, eval_data, eval_label):
    clf = svm.SVC(C=1, kernel='rbf', gamma=10, decision_function_shape='ovr')
    clf.fit(train_data, train_label)
    return clf.predict(eval_data)


def result_analysis(t_svm):
    """analysis the predicted result from classifiers for each descriptor, choose the most votes as final results"""
    col = t_svm.shape[1]
    svm_pre = np.empty(col)
    for i in range(col):
        svm = np.asarray(np.unique(t_svm[:, i], return_counts=True)).T
        svm_pre.put(i, svm[np.argmax(svm, axis=0), 0])
    return svm_pre


def crop_main(rgb_frame_feature_list, u_frame_feature_list, v_frame_feature_list, ucf_resNet_train_path,
              ucf_resNet_test_path, train_test_splits_save_path, dims, seeds,
              dataset='ucf'):
    if dataset == 'hmdb':
        tts = TrainTestSampleGen(ucf_path='', hmdb_path=train_test_splits_save_path)
    else:
        tts = TrainTestSampleGen(ucf_path=train_test_splits_save_path, hmdb_path='')

    encoder = preprocessing.LabelEncoder()
    # chosen split i
    for i in range(3,4):
        j = 0
        max_rgb = []
        max_u = []
        max_v = []
        for m in seeds:
            for rgb, u, v in zip(rgb_frame_feature_list, u_frame_feature_list, v_frame_feature_list):
                t_rgb, t_u, t_v, t_label = input_gen_all(rgb, u, v,
                                                         tts.train_data_label[i]['data'],
                                                         encoder.fit_transform(tts.train_data_label[i]['label']),
                                                         m, steps=train_steps, dim=dims)
                e_rgb, e_u, e_v, e_label = input_gen_all(rgb, u, v,
                                                         tts.test_data_label[i]['data'],
                                                         encoder.fit_transform(tts.test_data_label[i]['label']),
                                                         m, steps=train_steps, dim=dims)
                max_rgb.append(find_max(t_rgb, e_rgb))
                max_u.append(find_max(t_u, e_u))
                max_v.append(find_max(t_v, e_v))
                tf_record_writer(ucf_resNet_train_path, "rgb_flow_labels_" + str(j) + ".tfrecord", t_rgb,
                                 np.stack([t_u, t_v], axis=1),
                                 t_label)
                tf_record_writer(ucf_resNet_test_path, "rgb_flow_labels_" + str(j) + ".tfrecord", e_rgb,
                                 np.stack([e_u, e_v], axis=1), e_label)
                print("Version", j, "train-test data generation done")
                j += 1

        max_rgb = np.max(max_rgb, axis=0)
        max_u = np.max(max_u, axis=0)
        max_v = np.max(max_v, axis=0)
        return np.array([max_rgb, max_u, max_v]).astype(dtype=np.float32), encoder.fit_transform(
            tts.train_data_label[i]['label'])


if __name__ == '__main__':
    ucf_resNet_train_path_v1 = "/home/boy2/ucf101/ucf101_dataset/video_des/resNet152+Gaussian_train"
    ucf_resNet_test_path_v1 = "/home/boy2/ucf101/ucf101_dataset/video_des/resNet152+Gaussian_test"

    ucf_train_test_splits_save_path = "/home/boy2/ucf101/ucf101_dataset/ucfTrainTestlist"


    lto = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/left_top_o/rgb",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/left_top_o/u",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/left_top_o/v"]
    ltf = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/left_top_f/rgb",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/left_top_f/u",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/left_top_f/v"]
    lbo = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/left_bottom_o/rgb",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/left_bottom_o/u",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/left_bottom_o/v"]
    lbf = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/left_bottom_f/rgb",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/left_bottom_f/u",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/left_bottom_f/v"]
    rto = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/right_top_o/rgb",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/right_top_o/u",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/right_top_o/v"]
    rtf = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/right_top_f/rgb",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/right_top_f/u",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/right_top_f/v"]
    rbo = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/right_bottom_o/rgb",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/right_bottom_o/u",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/right_bottom_o/v"]
    rbf = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/right_bottom_f/rgb",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/right_bottom_f/u",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/right_bottom_f/v"]
    co = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/center_o/rgb",
          "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/center_o/u",
          "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/center_o/v"]
    cf = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/center_f/rgb",
          "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/center_f/u",
          "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/center_f/v"]

    # lto = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature_50/left_top_o/rgb",
    #        "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/left_top_o/u",
    #        "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/left_top_o/v"]
    # ltf = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature_50/left_top_f/rgb",
    #        "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/left_top_f/u",
    #        "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/left_top_f/v"]
    # lbo = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature_50/left_bottom_o/rgb",
    #        "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/left_bottom_o/u",
    #        "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/left_bottom_o/v"]
    # lbf = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature_50/left_bottom_f/rgb",
    #        "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/left_bottom_f/u",
    #        "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/left_bottom_f/v"]
    # rto = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature_50/right_top_o/rgb",
    #        "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/right_top_o/u",
    #        "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/right_top_o/v"]
    # rtf = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature_50/right_top_f/rgb",
    #        "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/right_top_f/u",
    #        "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/right_top_f/v"]
    # rbo = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature_50/right_bottom_o/rgb",
    #        "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/right_bottom_o/u",
    #        "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/right_bottom_o/v"]
    # rbf = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature_50/right_bottom_f/rgb",
    #        "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/right_bottom_f/u",
    #        "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/right_bottom_f/v"]
    # co = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature_50/center_o/rgb",
    #       "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/center_o/u",
    #       "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/center_o/v"]
    # cf = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature_50/center_f/rgb",
    #       "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/center_f/u",
    #       "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/center_f/v"]

    rgb = [e[0] for e in [lto, ltf, lbo, lbf, rto, rtf, rbo, rbf, co, cf]]
    u = [e[1] for e in [lto, ltf, lbo, lbf, rto, rtf, rbo, rbf, co, cf]]
    v = [e[2] for e in [lto, ltf, lbo, lbf, rto, rtf, rbo, rbf, co, cf]]

    # remove_dirctories([ucf_resNet_crop_ws_save_path_v1, ucf_resNet_crop_flow_ws_save_path_1_v1,
    #                    ucf_resNet_crop_flow_ws_save_path_2_v1,
    #                    ucf_resNet_crop_ws_save_path_v2, ucf_resNet_crop_flow_ws_save_path_1_v2,
    #                    ucf_resNet_crop_flow_ws_save_path_2_v2])

    best_acc = 0
    acc_list = []
    seed_list = []
    loc_list = []
    tran_m_list = []
    tran_m_list_norm = []
    # for i in range(int(num_trans_matrices)):
    #     seed = np.random.randint(1, 100)
    #     # seed_list.append(seed)
    #     print("The seed for trans_matrix is:", seed)
    #     tran_m = Weightedsum('tran_m', [], None).transformation_matrix_gen(dims, 1776, seed)
    #     tran_m_list.append(tran_m)
    #     tran_m_list_norm.append(preprocessing.normalize(tran_m, axis=1, norm='l2'))

    # 高斯
    # s = np.random.randint(0, dims, size=[num_trans_matrices-int(num_trans_matrices/2), int(np.floor((dims-1)/2))])
    loc = np.random.uniform(-1, 1, size=[int(num_trans_matrices), dims-1])
    # loc.sort()
    # for i in range(num_trans_matrices-int(num_trans_matrices/2)):
    #     seed_list.append(np.array([s[i], loc[i]]))

    for s in loc:
        tran_m_list_norm.append(s)

    # tran_m_list_norm = loc

    # tran_m_list_norm = np.array(tran_m_list_norm)
    np.save('/home/boy2/ucf101/ucf101_dataset/seed/seed.npy', tran_m_list_norm)
    # np.save('/home/boy2/ucf101/ucf101_dataset/frame_features/tras_m_1.npy', tran_m_list)
    # np.save('/home/boy2/ucf101/ucf101_dataset/frame_features/tras_m_1_norm.npy', tran_m_list_norm)

    # this = time()
    # max_value, test_gt = crop_main(
    #     rgb[:],
    #     u[:],
    #     v[:],
    #     ucf_resNet_train_path_v1, ucf_resNet_test_path_v1,
    #     ucf_train_test_splits_save_path, dims, tran_m_list_norm)
    # print(time() - this)
    # np.save(os.path.join(ucf_resNet_train_path_v1, "max_value.npy"), max_value)
    accuracy = classify(
        # os.path.join(ucf_resNet_train_path_v1, "rgb_flow_labels.tfrecord"),
        [os.path.join(ucf_resNet_train_path_v1, "rgb_flow_labels_" + str(i) + ".tfrecord") for i in
         range(num_samples_per_training_video * num_trans_matrices)],
        [os.path.join(ucf_resNet_test_path_v1, "rgb_flow_labels_" + str(i) + ".tfrecord") for i in
         range(num_samples_per_testing_video * num_trans_matrices)],
        num_train_data * num_samples_per_training_video * num_trans_matrices,
        num_test_data * num_samples_per_testing_video * num_trans_matrices,
        num_samples_per_testing_video,
        num_trans_matrices,
        (1, 2048, dims), (2, 2048, dims), os.path.join(ucf_resNet_train_path_v1, "max_value.npy"))
    print("accuracy for the current experiment is", accuracy)
    acc_list.append(accuracy)
    if accuracy > best_acc:
        best_acc = accuracy
    print("Accuracy for each exp is:", acc_list)
    np.savetxt('/home/boy2/ucf101/ucf101_dataset/result/result.txt', acc_list, delimiter=',')
