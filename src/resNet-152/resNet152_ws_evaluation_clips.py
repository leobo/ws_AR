import os

import numpy as np
import tensorflow as tf
from sklearn import preprocessing

from classifier.oned_conv_classifier_clips import classify_original_flips, classify_tfrecord, classify
from npyFileReader import Npyfilereader
from trainTestSamplesGen import TrainTestSampleGen
from weighted_sum import calWeightedSum

import re


def load_data(names, path):
    if type(names) is not str:
        return [np.load(os.path.join(path, name + '.npy')) for name in names]
    else:
        return np.load(os.path.join(path, names + '.npy'))


def split_and_store(name_list, image_feature_path, flow_feature_path_u, flow_feature_path_v,
                    store_path, clip_len=16, dim=2, ws=True, flip=False):
    i = 0
    full_path_list = []
    full_label_list = []
    for name, label in zip(name_list['data'], name_list['label']):
        rgb = load_data(name, image_feature_path)
        u = load_data(name, flow_feature_path_u)
        v = load_data(name, flow_feature_path_v)
        if flip == True:
            rgb = np.flip(rgb, 0)
            u = np.flip(u, 0)
            v = np.flip(v, 0)
        rgb, rgb_label_list, rgb_len = create_video_clips([rgb], [label], clip_len=clip_len)
        u = create_video_clips([u], rgb_len=rgb_len, clip_len=clip_len)
        v = create_video_clips([v], rgb_len=rgb_len, clip_len=clip_len)
        if ws == True:
            rgb = calWeightedSum.calculate_weightedsum_fixed_len(rgb, dim, clip_len)
            u = calWeightedSum.calculate_weightedsum_fixed_len(u, dim, clip_len)
            v = calWeightedSum.calculate_weightedsum_fixed_len(v, dim, clip_len)
            data = np.stack((rgb, u, v), axis=1)
        else:
            data = np.stack((rgb, u, v), axis=-1)
        for d, l in zip(data, rgb_label_list):
            path = os.path.join(store_path, str(i))
            np.save(path, {l: d})
            full_path_list.append(path)
            full_label_list.append(l)
            i += 1
    np.save(os.path.join(store_path, 'path_label_list'), {'data': full_path_list, 'label': full_label_list})
    return {'data': full_path_list, 'label': np.array(full_label_list)}


def create_video_clips(data, label=None, rgb_len=None, clip_len=16):
    data_clips = []
    if label is not None and rgb_len is None:
        label_clips = []
        rgb_len = []
        for d, l in zip(data, label):
            clip = [d[i: i + clip_len] if i + clip_len < len(d) else d[len(d) - clip_len: len(d)] for i
                    in range(0, len(d) - int(clip_len / 2), int(clip_len / 2))]
            data_clips += clip
            label_clips += [l for i in range(len(clip))]
            rgb_len.append(len(clip))
        return np.array(data_clips), np.array(label_clips), np.array(rgb_len)
    else:
        for d, l in zip(data, rgb_len):
            clip = [d[i: i + clip_len] if i + clip_len < len(d) else d[len(d) - clip_len: len(d)] for i
                    in range(0, len(d) - int(clip_len / 2), int(clip_len / 2))]
            if len(clip) < l:
                clip.append(clip[-1])
            data_clips += clip
        return np.array(data_clips)


def sort_numerically(file_list):
    """
    Sort the given list in the way that humans expect.
    :param file_list: the list contains file names
    :return: the numerically sorted file name list
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    file_list.sort(key=alphanum_key)
    return file_list


def read_data_label(save_path):
    data = []
    labels = []
    reader = Npyfilereader(save_path)
    reader.npy_paths = sort_numerically(reader.npy_paths)
    while len(reader.npy_paths) != 0:
        _, contents = reader.read_npys()
        if _ != "path_label_list.npy":
            contents = contents.item()
            for key, value in contents.items():
                data.append(value)
                labels.append(key)
    # encoder = preprocessing.LabelEncoder()
    # labels = encoder.fit_transform(labels)
    return {'data': np.array(data), 'label': np.array(labels)}


def read_path_label(path):
    return np.load(os.path.join(path, 'path_label_list' + '.npy')).item()


# if __name__ == '__main__':
#     np.save('/home/boy2/UCF101/ucf101_dataset/features/temp/path_label_list', {"data": [[1,2,3], [4,5,6]], "label": [1,2]})
#     a = read_path_label("/home/boy2/UCF101/ucf101_dataset/features/temp")
#     print()


def main(image_path, flow_path_u, flow_path_v, image_path_crop, flow_path_u_crop, flow_path_v_crop,
         train_test_splits_save_path, temp_train_store_path, temp_test_store_path, temp_train_store_path_crop,
         temp_test_store_path_crop, temp_train_store_path_flip, temp_test_store_path_flip,
         temp_train_store_path_flip_crop, temp_test_store_path_flip_crop, dataset='ucf'):
    if dataset == 'hmdb':
        tts = TrainTestSampleGen(ucf_path='', hmdb_path=train_test_splits_save_path)
    else:
        tts = TrainTestSampleGen(ucf_path=train_test_splits_save_path, hmdb_path='')
    acc = 0
    for i in range(1):
        train_list = split_and_store(tts.ucf_train_data_label[i], image_path, flow_path_u,
                                          flow_path_v, temp_train_store_path, 25)
        test_list = split_and_store(tts.ucf_test_data_label[i], image_path, flow_path_u,
                                         flow_path_v, temp_test_store_path, 25)

        train_list = read_data_label(temp_train_store_path)
        test_list = read_data_label(temp_test_store_path)

        encoder = preprocessing.LabelEncoder()
        train_list['label'] = encoder.fit_transform(train_list['label'])
        test_list['label'] = encoder.transform(test_list['label'])

        acc += classify(train_list, test_list)
    print('accuracy for ', dataset, 'is', acc / (i + 1))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_npy_to_tfrecord(npy_path, tfrecord_path):
    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    reader = Npyfilereader(npy_path)
    while len(reader.npy_paths) != 0:
        _, contents = reader.read_npys()
        if _ != "path_label_list.npy":
            contents = contents.item()
            for key, value in contents.items():
                feature = {'train/label': _int64_feature(key),
                           'train/feature': _bytes_feature(value.tostring())}
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())
    writer.close()


def temp(train_record, train_crop_record, test_record, test_crop_record):
        classify_tfrecord(train_record, train_crop_record, test_record, test_crop_record)


if __name__ == '__main__':
    ucf_resNet_flow_crop_save_path_1 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow_crop/u"
    ucf_resNet_flow_crop_save_path_2 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow_crop/v"
    ucf_resNet_crop_save_path = "/home/boy2/UCF101/ucf101_dataset/features/resNet_crop"

    ucf_resNet_flow_save_path_1 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow/u"
    ucf_resNet_flow_save_path_2 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow/v"
    ucf_resNet_save_path = "/home/boy2/UCF101/ucf101_dataset/features/resNet"

    ucf_train_test_splits_save_path = "/home/boy2/UCF101/ucf101_dataset/features/testTrainSplits"

    temp_train = "/home/boy2/UCF101/ucf101_dataset/features/temp/train"
    temp_test = "/home/boy2/UCF101/ucf101_dataset/features/temp/test"
    temp_train_crop = "/home/boy2/UCF101/ucf101_dataset/features/temp/train_crop"
    temp_test_crop = "/home/boy2/UCF101/ucf101_dataset/features/temp/test_crop"
    temp_train_flip = "/home/boy2/UCF101/ucf101_dataset/features/temp/train_flip"
    temp_test_flip = "/home/boy2/UCF101/ucf101_dataset/features/temp/test_flip"
    temp_train_flip_crop = "/home/boy2/UCF101/ucf101_dataset/features/temp/train_flip_crop"
    temp_test_flip_crop = "/home/boy2/UCF101/ucf101_dataset/features/temp/test_flip_crop"

    # convert_npy_to_tfrecord(temp_train, "/home/boy2/UCF101/ucf101_dataset/features/temp/train.tfrecord")
    # convert_npy_to_tfrecord(temp_test, "/home/boy2/UCF101/ucf101_dataset/features/temp/test.tfrecord")
    # convert_npy_to_tfrecord(temp_train_crop, "/home/boy2/UCF101/ucf101_dataset/features/temp/train_crop.tfrecord")
    # convert_npy_to_tfrecord(temp_test_crop, "/home/boy2/UCF101/ucf101_dataset/features/temp/test_crop.tfrecord")
    # convert_npy_to_tfrecord(temp_train_flip, "/home/boy2/UCF101/ucf101_dataset/features/temp/train_flip.tfrecord")
    # convert_npy_to_tfrecord(temp_test_flip, "/home/boy2/UCF101/ucf101_dataset/features/temp/test_flip.tfrecord")
    # convert_npy_to_tfrecord(temp_train_flip_crop,
    #                         "/home/boy2/UCF101/ucf101_dataset/features/temp/train_flip_crop.tfrecord")
    # convert_npy_to_tfrecord(temp_test_flip_crop,
    #                         "/home/boy2/UCF101/ucf101_dataset/features/temp/test_flip_crop.tfrecord")

    main(ucf_resNet_save_path, ucf_resNet_flow_save_path_1, ucf_resNet_flow_save_path_2, ucf_resNet_crop_save_path,
         ucf_resNet_flow_crop_save_path_1, ucf_resNet_flow_crop_save_path_2, ucf_train_test_splits_save_path,
         temp_train, temp_test, temp_train_crop, temp_test_crop, temp_train_flip, temp_test_flip, temp_train_flip_crop,
         temp_test_flip_crop)

    # temp("/home/boy2/UCF101/ucf101_dataset/features/temp/train.tfrecord",
    #      "/home/boy2/UCF101/ucf101_dataset/features/temp/train_crop.tfrecord",
    #      "/home/boy2/UCF101/ucf101_dataset/features/temp/test.tfrecord",
    #      "/home/boy2/UCF101/ucf101_dataset/features/temp/test_crop.tfrecord")
