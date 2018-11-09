import os

import numpy as np
import tensorflow as tf
from sklearn import preprocessing

from full_clips_classification.classifier import classify
from trainTestSamplesGen import TrainTestSampleGen

clip_length = 5
num_cha = 9
min_len_video = 30
max_len_video = 1000
num_train_data = 9537
num_test_data = 3783
# num_train_data = 10
# num_test_data = 10
# num_train = np.random.randint(0, 3783, size=10)
# num_test = np.random.randint(0, 3783, size=10)
num_trans_matrices = 1
num_frame_features = 1
num_sample_train = 1
num_sample_test = 25


def gen_subvideo_v2(rgb, u, v, steps):
    if len(rgb) == len(u) + 1:
        rgb = rgb[:-1]
    data_splits_indeces = np.array_split(np.arange(len(rgb)), steps)
    rgb_indeces = [np.random.choice(d) for d in data_splits_indeces]
    rgb_splits = rgb[rgb_indeces]
    flow_indeces = []
    for i in rgb_indeces:
        if i - 5 < 0:
            start = 0
            stop = 10
        elif i + 5 > len(rgb) - 1:
            stop = len(rgb)
            start = stop - 10
        else:
            start = i - 5
            stop = i + 5
        flow_indeces.append((start, stop))

    u_splits = np.array([u[index[0]:index[1]] for index in flow_indeces])
    v_splits = np.array([v[index[0]:index[1]] for index in flow_indeces])
    u_splits = np.reshape(u_splits, (u_splits.shape[0]*u_splits.shape[1], u_splits.shape[2]))
    v_splits = np.reshape(v_splits, (v_splits.shape[0] * v_splits.shape[1], v_splits.shape[2]))

    return np.transpose(rgb_splits), np.transpose(u_splits), np.transpose(v_splits)


def gen_subvideo(data, steps):
    data_par = np.array_split(data, steps)
    return np.array([d[np.random.randint(len(d))] for d in data_par])


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


def find_max(data):
    # normalize the data in time direction
    m = []
    if len(data.shape) == 3:
        x, y, t = data.shape
        for i in range(t):
            m.append(np.max(data[:, :, i], axis=0))
    else:
        temp_norm = preprocessing.normalize(data, axis=1, norm='max')
    return np.max(m, axis=0)


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


def input_gen_all(rgb_feature_path, u_feature_path, v_feature_path, video_names,
                  video_labels, steps=5, train=False):
    video_names = np.array(video_names)
    video_labels = video_labels[:num_train_data]
    video_names = video_names[:num_train_data]

    # video_labels = video_labels[num_train]
    # video_names = video_names[num_train]
    # 1776 the max len video
    rgb_path = [os.path.join(rgb_feature_path, n + '.npy') for n in video_names]
    u_path = [os.path.join(u_feature_path, n + '.npy') for n in video_names]
    v_path = [os.path.join(v_feature_path, n + '.npy') for n in video_names]
    all_rgb = []
    all_u = []
    all_v = []
    all_label = []

    for rp, up, vp, name, l in zip(rgb_path, u_path, v_path, video_names,
                                   video_labels):
        rgb = np.load(rp)
        u = np.load(up)
        v = np.load(vp)

        rgb, u, v = gen_subvideo_v2(rgb, u, v, steps)

        # rgb = np.transpose(gen_subvideo(rgb, int(steps)))
        # u = np.transpose(gen_subvideo(u, steps))
        # v = np.transpose(gen_subvideo(v, steps))

        all_rgb.append(rgb)
        all_u.append(u)
        all_v.append(v)
        all_label.append(l)

    return np.array(all_rgb), np.array(all_u), np.array(all_v), np.array(all_label)


# def input_gen_all_test(rgb_feature_path, u_feature_path, v_feature_path, selected_frames_path, video_names,
#                        video_labels,
#                        seed, steps=5, dim=2, num_samples=5):
#     video_names = np.array(video_names)
#     video_labels = video_labels[:num_train_data]
#     video_names = video_names[:num_train_data]
#
#     # video_labels = video_labels[num_train]
#     # video_names = video_names[num_train]
#     # 1776 the max len video
#     rgb_path = [os.path.join(rgb_feature_path, n + '.npy') for n in video_names]
#     u_path = [os.path.join(u_feature_path, n + '.npy') for n in video_names]
#     v_path = [os.path.join(v_feature_path, n + '.npy') for n in video_names]
#     selected_frames = [os.path.join(selected_frames_path, n + '.mat') for n in video_names]
#     all_rgb = []
#     all_u = []
#     all_v = []
#     all_label = []
#
#     for rp, up, vp, sf, name, l in zip(rgb_path, u_path, v_path, selected_frames, video_names,
#                                        video_labels):
#         rgb = np.load(rp)
#         u = np.load(up)
#         v = np.load(vp)
#
#         if len(seed.shape) == 1:
#             tran_m_rgb = Weightedsum('tran_m', [], None).transformation_matrix_gen_norm(dims, len(rgb), seed)
#             tran_m_rgb = preprocessing.normalize(tran_m_rgb, axis=1, norm='l2')
#             tran_m_flow = Weightedsum('tran_m', [], None).transformation_matrix_gen_norm(dims, len(u), seed)
#             tran_m_flow = preprocessing.normalize(tran_m_flow, axis=1, norm='l2')
#         else:
#             tran_m_rgb = seed
#             tran_m_flow = seed
#
#         for i in range(num_samples):
#             rgb = gen_subvideo(rgb, steps)
#             u = gen_subvideo(u, steps)
#             v = gen_subvideo(v, steps)
#
#             _rgb = Weightedsum(name, rgb[:], None).ws_descriptor_gen(dim, False, tran_m_rgb)
#             _u = Weightedsum(name, u[:], None).ws_descriptor_gen(dim, False, tran_m_flow)
#             _v = Weightedsum(name, v[:], None).ws_descriptor_gen(dim, False, tran_m_flow)
#
#             all_rgb.append(_rgb)
#             all_u.append(_u)
#             all_v.append(_v)
#             all_label.append(l)
#
#     return all_rgb, all_u, all_v, all_label


# def input_gen_all_level(rgb_feature_path, u_feature_path, v_feature_path, selected_frames_path, video_names,
#                         video_labels,
#                         seed, steps=5, dim=2, train=False):
#     video_names = np.array(video_names)
#     video_labels = video_labels[:num_train_data]
#     video_names = video_names[:num_train_data]
#
#     # video_labels = video_labels[num_train]
#     # video_names = video_names[num_train]
#     # 1776 the max len video
#     rgb_path = [os.path.join(rgb_feature_path, n + '.npy') for n in video_names]
#     u_path = [os.path.join(u_feature_path, n + '.npy') for n in video_names]
#     v_path = [os.path.join(v_feature_path, n + '.npy') for n in video_names]
#     selected_frames = [os.path.join(selected_frames_path, n + '.mat') for n in video_names]
#     all_rgb = []
#     all_u = []
#     all_v = []
#     all_label = []
#
#     conv_depth = 1
#     first_c = 4
#     second_c = 4
#
#     for rp, up, vp, sf, name, l in zip(rgb_path, u_path, v_path, selected_frames, video_names,
#                                        video_labels):
#         rgb = np.load(rp)
#         u = np.load(up)
#         v = np.load(vp)
#
#         for i in range(conv_depth):
#             rgb = np.array_split(rgb, steps)
#             u = np.array_split(u, steps)
#             v = np.array_split(v, steps)
#
#         rgb = np.array(rgb)
#         u = np.array(u)
#         v = np.array(v)
#
#         rgb_temp = []
#         u_temp = []
#         v_temp = []
#
#         for _r, _u, _v in zip(rgb, u, v):
#             if len(seed.shape) == 1:
#                 tran_m_rgb = Weightedsum('tran_m', [], None).transformation_matrix_gen_norm(dims, len(_r), seed)
#                 tran_m_rgb = preprocessing.normalize(tran_m_rgb, axis=1, norm='l2')
#                 tran_m_flow = Weightedsum('tran_m', [], None).transformation_matrix_gen_norm(dims, len(_u), seed)
#                 tran_m_flow = preprocessing.normalize(tran_m_flow, axis=1, norm='l2')
#             else:
#                 tran_m_rgb = seed
#                 tran_m_flow = seed
#
#             _rgb = Weightedsum(name, _r, None).ws_descriptor_gen(dim, False, tran_m_rgb[:first_c])
#             _u = Weightedsum(name, _u, None).ws_descriptor_gen(dim, False, tran_m_flow[:first_c])
#             _v = Weightedsum(name, _v, None).ws_descriptor_gen(dim, False, tran_m_flow[:first_c])
#
#             rgb_temp.append(_rgb)
#             u_temp.append(_u)
#             v_temp.append(_v)
#
#         rgb = np.array(rgb_temp)
#         u = np.array(u_temp)
#         v = np.array(v_temp)
#
#         rgb_temp = []
#         u_temp = []
#         v_temp = []
#
#         for i in range(np.shape(rgb)[-1]):
#             if len(seed.shape) == 1:
#                 tran_m_rgb = Weightedsum('tran_m', [], None).transformation_matrix_gen_norm(dims, steps, seed)
#                 tran_m_rgb = preprocessing.normalize(tran_m_rgb, axis=1, norm='l2')
#                 tran_m_flow = Weightedsum('tran_m', [], None).transformation_matrix_gen_norm(dims, steps, seed)
#                 tran_m_flow = preprocessing.normalize(tran_m_flow, axis=1, norm='l2')
#             else:
#                 tran_m_rgb = seed
#                 tran_m_flow = seed
#
#             _rgb = Weightedsum(name, rgb[:, :, i], None).ws_descriptor_gen(dim, False, tran_m_rgb[:second_c])
#             _u = Weightedsum(name, u[:, :, i], None).ws_descriptor_gen(dim, False, tran_m_flow[:second_c])
#             _v = Weightedsum(name, v[:, :, i], None).ws_descriptor_gen(dim, False, tran_m_flow[:second_c])
#
#             rgb_temp.append(_rgb)
#             u_temp.append(_u)
#             v_temp.append(_v)
#
#         rgb = np.array(rgb_temp)
#         u = np.array(u_temp)
#         v = np.array(v_temp)
#
#         rgb = np.swapaxes(rgb, 1, 0)
#         u = np.swapaxes(u, 1, 0)
#         v = np.swapaxes(v, 1, 0)
#
#         rgb = np.reshape(rgb, [np.shape(rgb)[0], np.shape(rgb)[1] * np.shape(rgb)[-1]])
#         u = np.reshape(u, [np.shape(u)[0], np.shape(u)[1] * np.shape(u)[-1]])
#         v = np.reshape(v, [np.shape(v)[0], np.shape(v)[1] * np.shape(v)[-1]])
#
#         all_rgb.append(rgb)
#         all_u.append(u)
#         all_v.append(v)
#         all_label.append(l)
#
#     return all_rgb, all_u, all_v, all_label
#
#
# def input_gen_mul(rgb_feature_path, u_feature_path, v_feature_path, video_names, video_labels, steps=5, dim=2,
#                   train=False):
#     # 1775 the max len video
#     rgb_path = [os.path.join(rgb_feature_path, n + '.npy') for n in video_names][:]
#     u_path = [os.path.join(u_feature_path, n + '.npy') for n in video_names][:]
#     all_rgb = []
#     all_u = []
#     all_label = []
#     tran_m = Weightedsum('tran_m', [], None).transformation_matrix_gen(dim, 1776)
#     tran_m = preprocessing.normalize(tran_m, axis=1)
#     # tran_m = None
#     for rp, up, name, l in zip(rgb_path, u_path, video_names[:], video_labels[:]):
#         if rp != "/home/boy2/UCF101/ucf101_dataset/features/mulNet_feature/mulNet/rgb/o/v_StillRings_g21_c05.npy":
#             rgb = np.load(rp)
#             rgb = np.transpose(rgb)
#             u = np.load(up)
#             u = np.transpose(u)
#             if len(rgb) > len(u) and len(rgb) - 1 == len(u):
#                 rgb = rgb[:-1]
#
#             # # Time Varying Mean Vectors
#             # copy_rgb = rgb.copy()
#             # copy_u = u.copy()
#             # copy_v = v.copy()
#             # for r in range(len(rgb)):
#             #     rgb[r] = np.sum(copy_rgb[:r + 1, :], axis=0) / (r + 1)
#             #     rgb[r] = rgb[r] / np.linalg.norm(rgb[r])
#             #     u[r] = np.sum(copy_u[:r + 1, :], axis=0) / (r + 1)
#             #     u[r] = u[r] / np.linalg.norm(u[r])
#             #     v[r] = np.sum(copy_v[:r + 1, :], axis=0) / (r + 1)
#             #     v[r] = v[r] / np.linalg.norm(v[r])
#
#             _rgb = Weightedsum(name, rgb[:], None).ws_descriptor_gen(dim, False, tran_m)
#             _u = Weightedsum(name, u[:], None).ws_descriptor_gen(dim, False, tran_m)
#
#             all_rgb.append(_rgb / len(rgb))
#             all_u.append(_u / len(u))
#             all_label.append(l)
#     return all_rgb, all_u, all_label
#
#
# def prdict(train_data, train_label, eval_data, eval_label):
#     clf = svm.SVC(C=1, kernel='rbf', gamma=10, decision_function_shape='ovr')
#     clf.fit(train_data, train_label)
#     return clf.predict(eval_data)
#
#
# def result_analysis(t_svm):
#     """analysis the predicted result from classifiers for each descriptor, choose the most votes as final results"""
#     col = t_svm.shape[1]
#     svm_pre = np.empty(col)
#     for i in range(col):
#         svm = np.asarray(np.unique(t_svm[:, i], return_counts=True)).T
#         svm_pre.put(i, svm[np.argmax(svm, axis=0), 0])
#     return svm_pre


def crop_main(rgb_frame_feature_list, u_frame_feature_list, v_frame_feature_list, ucf_resNet_train_path,
              ucf_resNet_test_path, train_test_splits_save_path, clip_length, num_test_sample,
              dataset='ucf'):
    if dataset == 'hmdb':
        tts = TrainTestSampleGen(ucf_path='', hmdb_path=train_test_splits_save_path)
    else:
        tts = TrainTestSampleGen(ucf_path=train_test_splits_save_path, hmdb_path='')

    encoder = preprocessing.LabelEncoder()
    for i in range(1):
        j = 0
        k = 0
        max_rgb = []
        max_u = []
        max_v = []
        for rgb, u, v in zip(rgb_frame_feature_list, u_frame_feature_list, v_frame_feature_list):
            for _ in range(num_sample_train):
                t_rgb, t_u, t_v, t_label = input_gen_all(rgb, u, v,
                                                         tts.train_data_label[i]['data'],
                                                         encoder.fit_transform(tts.train_data_label[i]['label']),
                                                         steps=clip_length)
                max_rgb.append(find_max(t_rgb))
                max_u.append(find_max(t_u))
                max_v.append(find_max(t_v))

                tf_record_writer(ucf_resNet_train_path, "rgb_flow_labels_" + str(j) + ".tfrecord", t_rgb,
                                 np.stack([t_u, t_v], axis=1),
                                 t_label)
                print("Version", j, "train data generation done")
                j += 1

            for _ in range(num_sample_test):
                e_rgb, e_u, e_v, e_label = input_gen_all(rgb, u, v,
                                                         tts.test_data_label[i]['data'],
                                                         encoder.fit_transform(tts.test_data_label[i]['label']),
                                                         steps=clip_length)
                max_rgb.append(find_max(e_rgb))
                max_u.append(find_max(e_u))
                max_v.append(find_max(e_v))
                tf_record_writer(ucf_resNet_test_path, "rgb_flow_labels_" + str(k) + ".tfrecord", e_rgb,
                                 np.stack([e_u, e_v], axis=1), e_label)
                print("Version", k, "test data generation done")
                k += 1

        max_rgb = np.max(max_rgb, axis=0)
        max_u = np.max(max_u, axis=0)
        max_v = np.max(max_v, axis=0)
        return np.array([max_rgb, max_u, max_v]).astype(dtype=np.float32), encoder.fit_transform(
            tts.train_data_label[i]['label'])


if __name__ == '__main__':
    ucf_resNet_flow_crop_save_path_1_v4 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop_v4/u"
    ucf_resNet_flow_crop_save_path_2_v4 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop_v4/v"
    ucf_resNet_crop_save_path_v4 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_crop_v4"

    ucf_resNet_flow_crop_save_path_1_v3 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop_v3/u"
    ucf_resNet_flow_crop_save_path_2_v3 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop_v3/v"
    ucf_resNet_crop_save_path_v3 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_crop_v3"

    ucf_resNet_flow_crop_save_path_1_v2 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop_v2/u"
    ucf_resNet_flow_crop_save_path_2_v2 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop_v2/v"
    ucf_resNet_crop_save_path_v2 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_crop_v2"

    ucf_resNet_flow_crop_save_path_1_v1 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop/u"
    # ucf_resNet_flow_crop_save_path_1_v1 = "/home/boy2/UCF101/ucf101_dataset/features/mulNet_feature/mulNet/flow/o"
    ucf_resNet_flow_crop_save_path_2_v1 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop/v"
    ucf_resNet_crop_save_path_v1 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_crop"
    # ucf_resNet_crop_save_path_v1 = "/home/boy2/UCF101/ucf101_dataset/features/mulNet_feature/mulNet/rgb/o"

    ucf_resNet_flow_save_path_1 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow/u"
    ucf_resNet_flow_save_path_2 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow/v"
    ucf_resNet_save_path = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet"

    featureStorePath = ["/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_resize",
                        "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_resize/u",
                        "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_resize/v"]
    featureStorePathFlip = ["/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_resize_flip",
                            "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_resize_flip/u",
                            "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_resize_flip/v"]

    ucf_resNet_train_path_v1 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_train_v1"
    ucf_resNet_test_path_v1 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_test_v1"

    ucf_resNet_train_path_v2 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_train_v2"
    ucf_resNet_test_path_v2 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_test_v2"

    ucf_resNet_train_path_v3 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_train_v3"
    ucf_resNet_test_path_v3 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_test_v3"

    ucf_resNet_train_path_v4 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_train_v4"
    ucf_resNet_test_path_v4 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_test_v4"

    ucf_train_test_splits_save_path = "/home/boy2/UCF101/ucf101_dataset/features/testTrainSplits"

    selected_frames_path = '/home/boy2/UCF101/ucf101_dataset/frame_features/seq_number'

    lto = ["/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_top_o/rgb",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_top_o/u",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_top_o/v"]
    ltf = ["/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_top_f/rgb",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_top_f/u",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_top_f/v"]
    lbo = ["/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_bottom_o/rgb",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_bottom_o/u",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_bottom_o/v"]
    lbf = ["/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_bottom_f/rgb",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_bottom_f/u",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_bottom_f/v"]
    rto = ["/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_top_o/rgb",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_top_o/u",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_top_o/v"]
    rtf = ["/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_top_f/rgb",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_top_f/u",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_top_f/v"]
    rbo = ["/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_bottom_o/rgb",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_bottom_o/u",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_bottom_o/v"]
    rbf = ["/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_bottom_f/rgb",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_bottom_f/u",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_bottom_f/v"]
    co = ["/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/center_o/rgb",
          "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/center_o/u",
          "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/center_o/v"]
    cf = ["/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/center_f/rgb",
          "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/center_f/u",
          "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/center_f/v"]

    rgb = [e[0] for e in [lto, lbo, rto, rbo, co, ltf, lbf, rtf, rbf, cf]]
    u = [e[1] for e in [lto, lbo, rto, rbo, co, ltf, lbf, rtf, rbf, cf]]
    v = [e[2] for e in [lto, lbo, rto, rbo, co, ltf, lbf, rtf, rbf, cf]]

    # remove_dirctories([ucf_resNet_crop_ws_save_path_v1, ucf_resNet_crop_flow_ws_save_path_1_v1,
    #                    ucf_resNet_crop_flow_ws_save_path_2_v1,
    #                    ucf_resNet_crop_ws_save_path_v2, ucf_resNet_crop_flow_ws_save_path_1_v2,
    #                    ucf_resNet_crop_flow_ws_save_path_2_v2])

    best_acc = 0
    acc_list = []

    max_value, test_gt = crop_main(
        rgb[:num_frame_features],
        u[:num_frame_features],
        v[:num_frame_features],
        ucf_resNet_train_path_v1, ucf_resNet_test_path_v1,
        ucf_train_test_splits_save_path, clip_length, num_sample_test)
    np.save(os.path.join(ucf_resNet_train_path_v1, "max_value.npy"), max_value)
    accuracy = classify(
        # os.path.join(ucf_resNet_train_path_v1, "rgb_flow_labels.tfrecord"),
        [os.path.join(ucf_resNet_train_path_v1, "rgb_flow_labels_" + str(i) + ".tfrecord") for i in
         range(num_frame_features * num_sample_train * num_trans_matrices)],
        [os.path.join(ucf_resNet_test_path_v1, "rgb_flow_labels_" + str(i) + ".tfrecord") for i in
         range(num_frame_features * num_sample_test * num_trans_matrices)],
        num_train_data * num_frame_features * num_trans_matrices * num_sample_train,
        num_test_data * num_frame_features * num_trans_matrices * num_sample_test,
        num_frame_features * num_sample_test,
        num_trans_matrices,
        (1, 2048, clip_length), (2, 2048, clip_length*10), os.path.join(ucf_resNet_train_path_v1, "max_value.npy"))
    print("accuracy for the current experiment is", accuracy)
    acc_list.append(accuracy)
    if accuracy > best_acc:
        best_acc = accuracy
    print("Accuracy for each exp is:", acc_list)
    np.savetxt('/home/boy2/UCF101/ucf101_dataset/frame_features/result.txt', acc_list, delimiter=',')
