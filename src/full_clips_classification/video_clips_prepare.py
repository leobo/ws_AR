import os

import numpy as np
import tensorflow as tf
from sklearn import preprocessing

from full_clips_classification.video_clips_classifier import classify
from trainTestSamplesGen import TrainTestSampleGen

num_samples_per_training_video = 1
num_samples_per_testing_video = 1
train_steps = 25
test_steps = 25
dims = 2
min_len_video = 30
max_len_video = 1000
num_train_data = 9537
num_test_data = 3783


def norm_encode_data(train_data, test_data):
    len_train = len(train_data)
    # normalize the data in time direction
    temp = np.concatenate([train_data, test_data])
    if len(temp.shape) == 3:
        x, y, t = temp.shape
        temp_norm = np.zeros(shape=(x, y, t))
        for i in range(t):
            temp_norm[:, :, i] = preprocessing.normalize(temp[:, :, i], axis=1, norm='max')
    else:
        temp_norm = preprocessing.normalize(temp, axis=1)
    train_data = temp_norm[:len_train]
    test_data = temp_norm[len_train:]
    return train_data, test_data


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


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


def input_gen_all(rgb_feature_path, u_feature_path, v_feature_path, video_names, video_labels, tfrecord_path,
                  dim=2, clip_len=16, overlap=0.5, flip=False, num_samples=10, steps=5):
    rgb_path = [os.path.join(rgb_feature_path, n + '.npy') for n in video_names][:]
    u_path = [os.path.join(u_feature_path, n + '.npy') for n in video_names][:]
    v_path = [os.path.join(v_feature_path, n + '.npy') for n in video_names][:]

    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    _num = 0
    for rp, up, vp, name, l in zip(rgb_path, u_path, v_path, video_names[:],
                                   video_labels[:]):
        rgb = np.load(rp)
        u = np.load(up)
        v = np.load(vp)
        if len(rgb) - 1 == len(u):
            rgb = rgb[:-1]
        rgb_temp = rgb
        u_temp = u
        v_temp = v
        # stack video frames to make enough number of frames
        if len(u) < num_samples * steps:
            num = int(num_samples * steps / len(u)) + 1
            for i in range(num):
                rgb = np.vstack((rgb, rgb_temp))
                u = np.vstack((u, u_temp))
                v = np.vstack((v, v_temp))
        _num += 1
        print(rp)

        rgb_par = np.array_split(rgb, steps)
        u_par = np.array_split(u, steps)
        v_par = np.array_split(v, steps)
        for _ in range(num_samples):
            t_r = []
            t_u = []
            t_v = []
            for _r, _uf, _vf in zip(rgb_par, u_par, v_par):
                t_r.append(_r[_])
                t_u.append(_uf[_])
                t_v.append(_vf[_])

            # write the data into tfrecord
            feature = {'rgb': _bytes_feature(np.array(t_r).astype(np.float32).tobytes()),
                       'flow_u': _bytes_feature(np.array(t_u).astype(np.float32).tobytes()),
                       'flow_v': _bytes_feature(np.array(t_v).astype(np.float32).tobytes()),
                       'labels': _int64_feature(l)
                       }
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
    writer.close()
    return _num


def create_video_clips(data, label, clip_len=16, overlap=0.5):
    data_clips = []
    label_clips = []
    for d, l in zip(data, label):
        clip = []
        for i in range(0, len(d) - int(clip_len * (1 - overlap)), int(clip_len * (1 - overlap))):
            if i + clip_len < len(d):
                clip.append(d[i: i + clip_len])
            else:
                clip.append(d[len(d) - clip_len: len(d)])
                break
        data_clips += clip
        label_clips += [l for i in range(len(clip))]
    return np.array(data_clips), np.array(label_clips)


def crop_main(resNet_crop_save_path_v1, resNet_flow_crop_save_path_1_v1, resNet_flow_crop_save_path_2_v1,
              resNet_train_ws_v1, resNet_test_ws_v1,
              resNet_crop_save_path_v2, resNet_flow_crop_save_path_1_v2, resNet_flow_crop_save_path_2_v2,
              resNet_train_ws_v2, resNet_test_ws_v2,
              resNet_crop_save_path_v3, resNet_flow_crop_save_path_1_v3, resNet_flow_crop_save_path_2_v3,
              resNet_train_ws_v3, resNet_test_ws_v3,
              train_test_splits_save_path, dim, dataset='ucf'):
    if dataset == 'hmdb':
        tts = TrainTestSampleGen(ucf_path='', hmdb_path=train_test_splits_save_path)
    else:
        tts = TrainTestSampleGen(ucf_path=train_test_splits_save_path, hmdb_path='')

    accuracy = 0
    encoder = preprocessing.LabelEncoder()
    for i in range(1):
        # train-data
        num_train_data = input_gen_all(resNet_crop_save_path_v1, resNet_flow_crop_save_path_1_v1,
                                       resNet_flow_crop_save_path_2_v1,
                                       tts.train_data_label[i]['data'],
                                       encoder.fit_transform(tts.train_data_label[i]['label']),
                                       resNet_train_ws_v1,
                                       dim=dim, clip_len=25, overlap=0.5, flip=False,
                                       num_samples=num_samples_per_training_video,
                                       steps=train_steps)
        # test-data
        num_test_data = input_gen_all(resNet_crop_save_path_v1, resNet_flow_crop_save_path_1_v1,
                                      resNet_flow_crop_save_path_2_v1,
                                      tts.test_data_label[i]['data'],
                                      encoder.fit_transform(tts.test_data_label[i]['label']),
                                      resNet_test_ws_v1,
                                      dim=dim, clip_len=25, overlap=0.5, flip=False,
                                      num_samples=num_samples_per_testing_video,
                                      steps=test_steps)

        # _t_rgb, _t_u, _t_v, _t_label = input_gen_all(resNet_crop_save_path_v2, resNet_flow_crop_save_path_1_v2,
        #                                              resNet_flow_crop_save_path_2_v2,
        #                                              tts.ucf_train_data_label[i]['data'],
        #                                              encoder.fit_transform(tts.ucf_train_data_label[i]['label']),
        #                                              steps=train_steps, train=True, dim=dims)
        # _e_rgb, _e_u, _e_v, _e_label = input_gen_all(resNet_crop_save_path_v2, resNet_flow_crop_save_path_1_v2,
        #                                              resNet_flow_crop_save_path_2_v2,
        #                                              tts.ucf_test_data_label[i]['data'],
        #                                              encoder.fit_transform(tts.ucf_test_data_label[i]['label']),
        #                                              steps=test_steps, train=False, dim=dims)
        #
        # t_rgb3, t_u3, t_v3, t_label3 = input_gen_all(resNet_crop_save_path_v3, resNet_flow_crop_save_path_1_v3,
        #                                              resNet_flow_crop_save_path_2_v3,
        #                                              tts.ucf_train_data_label[i]['data'],
        #                                              encoder.fit_transform(tts.ucf_train_data_label[i]['label']),
        #                                              steps=train_steps, train=True, dim=dims)
        # e_rgb3, e_u3, e_v3, e_label3 = input_gen_all(resNet_crop_save_path_v3, resNet_flow_crop_save_path_1_v3,
        #                                              resNet_flow_crop_save_path_2_v3,
        #                                              tts.ucf_test_data_label[i]['data'],
        #                                              encoder.fit_transform(tts.ucf_test_data_label[i]['label']),
        #                                              steps=test_steps, train=False, dim=dims)

        accuracy += classify(
            resNet_train_ws_v1,
            resNet_test_ws_v1,
            num_train_data * num_samples_per_training_video,
            num_test_data * num_samples_per_testing_video,
            num_samples_per_testing_video,
            (train_steps, 2048, 1))
        print("accuracy is", accuracy)


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
    ucf_resNet_flow_crop_save_path_2_v1 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop/v"
    ucf_resNet_crop_save_path_v1 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_crop"

    ucf_resNet_flow_save_path_1 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow/u"
    ucf_resNet_flow_save_path_2 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow/v"
    ucf_resNet_save_path = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet"

    ucf_resNet_train_path_v1 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_train_v1/train_clips.tfrecord"
    ucf_resNet_test_path_v1 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_test_v1/test_clips.tfrecord"

    ucf_resNet_train_path_v2 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_train_v2"
    ucf_resNet_test_path_v2 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_test_v2"

    ucf_resNet_train_path_v3 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_train_v3"
    ucf_resNet_test_path_v3 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_test_v3"

    ucf_resNet_train_path_v4 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_train_v4"
    ucf_resNet_test_path_v4 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_test_v4"

    ucf_train_test_splits_save_path = "/home/boy2/UCF101/ucf101_dataset/features/testTrainSplits"

    # remove_dirctories([ucf_resNet_crop_ws_save_path_v1, ucf_resNet_crop_flow_ws_save_path_1_v1,
    #                    ucf_resNet_crop_flow_ws_save_path_2_v1,
    #                    ucf_resNet_crop_ws_save_path_v2, ucf_resNet_crop_flow_ws_save_path_1_v2,
    #                    ucf_resNet_crop_flow_ws_save_path_2_v2])

    crop_main(ucf_resNet_crop_save_path_v1, ucf_resNet_flow_crop_save_path_1_v1, ucf_resNet_flow_crop_save_path_2_v1,
              ucf_resNet_train_path_v1, ucf_resNet_test_path_v1,
              ucf_resNet_crop_save_path_v2, ucf_resNet_flow_crop_save_path_1_v2, ucf_resNet_flow_crop_save_path_2_v2,
              ucf_resNet_train_path_v3, ucf_resNet_test_path_v3,
              ucf_resNet_crop_save_path_v4, ucf_resNet_flow_crop_save_path_1_v4, ucf_resNet_flow_crop_save_path_2_v4,
              ucf_resNet_train_path_v4, ucf_resNet_test_path_v4,
              ucf_train_test_splits_save_path, dims)
