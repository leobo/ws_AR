import os
import re

import numpy as np
from sklearn import preprocessing

from trainTestSamplesGen import TrainTestSampleGen
from weighted_sum import calWeightedSum


def load_data(names, path):
    if type(names) is not str:
        return [np.load(os.path.join(path, name + '.npy')) for name in names]
    else:
        return np.load(os.path.join(path, names + '.npy'))


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


def create_video_clips_gap(data, label, gap=5):
    data_clips = []
    label_clips = []
    for d, l in zip(data, label):
        clip = []
        for i in range(gap):
            clip.append(d[i::gap])
        print()


if __name__ == '__main__':
    d = np.arange(104)
    d = np.reshape(d, newshape=(4, 26))
    create_video_clips_gap(d, d, gap=5)


def split_and_store(name_list, image_feature_path, flow_feature_path_u, flow_feature_path_v,
                    store_path, clip_len=16, dim=2, flip=False, ws=True, overlap=0.5):
    i = 0
    full_path_list = []
    full_label_list = []
    for name, label in zip(name_list['data'], name_list['label']):
        rgb = load_data(name, image_feature_path)
        u = load_data(name, flow_feature_path_u)
        v = load_data(name, flow_feature_path_v)
        if len(rgb) - 1 == len(u) == len(v):
            print('Remove last frame for video', name)
            rgb = rgb[:-1]
        elif len(rgb) != (len(u) or len(v)):
            print('!!!!!!!!!!!!!!!!!!!!The length of rgb with flow u and v are not correct', name)
            print('The length of rgb is', len(rgb), 'The length of flow u and v are', len(u), len(v))
            return None
        rgb, rgb_label_list = create_video_clips([rgb], [label], clip_len=clip_len, overlap=overlap)
        u, _ = create_video_clips([u], [label], clip_len=clip_len, overlap=overlap)
        v, _ = create_video_clips([v], [label], clip_len=clip_len, overlap=overlap)
        if ws == True:
            rgb = calWeightedSum.calculate_weightedsum_fixed_len(rgb, dim, clip_len, flip)
            u = calWeightedSum.calculate_weightedsum_fixed_len(u, dim, clip_len, flip)
            v = calWeightedSum.calculate_weightedsum_fixed_len(v, dim, clip_len, flip)
            for h in range(dim):
                rgb[:, :, h] = preprocessing.normalize(rgb[:, :, h], axis=1)
                u[:, :, h] = preprocessing.normalize(u[:, :, h], axis=1)
                v[:, :, h] = preprocessing.normalize(v[:, :, h], axis=1)
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


def creat():
    ucf_resNet_flow_crop_save_path_1 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop/u"
    ucf_resNet_flow_crop_save_path_2 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop/v"
    ucf_resNet_crop_save_path = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_crop"

    ucf_resNet_flow_save_path_1 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow/u"
    ucf_resNet_flow_save_path_2 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow/v"
    ucf_resNet_save_path = "/home/boy2/UCF101/ucf101_dataset/features/resNet"

    ucf_train_test_splits_save_path = "/home/boy2/UCF101/ucf101_dataset/features/testTrainSplits"

    temp_train_store_path = "/home/boy2/UCF101/ucf101_dataset/features/important_resNet_crop_ws_train"
    temp_test_store_path = "/home/boy2/UCF101/ucf101_dataset/features/important_resNet_crop_ws_test"

    tts = TrainTestSampleGen(ucf_path=ucf_train_test_splits_save_path, hmdb_path='')

    split_and_store(tts.ucf_train_data_label[0], ucf_resNet_crop_save_path, ucf_resNet_flow_crop_save_path_1,
                    ucf_resNet_flow_crop_save_path_2, temp_train_store_path, clip_len=25, dim=2, overlap=0.5)

    split_and_store(tts.ucf_test_data_label[0], ucf_resNet_crop_save_path, ucf_resNet_flow_crop_save_path_1,
                    ucf_resNet_flow_crop_save_path_2, temp_test_store_path, clip_len=25, dim=2, overlap=0.5)
