import os
import shutil
import tensorflow as tf

import numpy as np
from sklearn import preprocessing

from classifier import onedconv_classifier
from npyFileReader import Npyfilereader
from trainTestSamplesGen import TrainTestSampleGen
from weighted_sum import calWeightedSum
from weighted_sum.videoDescriptorWeightedSum import Weightedsum


def norm_encode_data(train_data_label_image, test_data_label_image, encoder):
    len_train = len(train_data_label_image['data'])
    # normalize the data in time direction
    temp = np.concatenate([train_data_label_image['data'], test_data_label_image['data']])
    if len(temp.shape) == 3:
        x, y, t = temp.shape
        temp_norm = np.zeros(shape=(x, y, t))
        for i in range(t):
            temp_norm[:, :, i] = preprocessing.normalize(temp[:, :, i], axis=1)
    else:
        temp_norm = preprocessing.normalize(temp, axis=1)
    # optional: normalize each sample data independently
    # temp_norm = preprocessing.normalize(temp_norm, axis=1)
    train_data_label_image['data'] = temp_norm[:len_train]
    test_data_label_image['data'] = temp_norm[len_train:]
    train_data_label_image['label'] = encoder.fit_transform(train_data_label_image['label']) + 1
    test_data_label_image['label'] = encoder.fit_transform(test_data_label_image['label']) + 1
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


def reformat_reshaped(data_label_image, data_label_flow_1, data_label_flow_2):
    data = []
    for i, f1, f2 in zip(data_label_image['data'], data_label_flow_1['data'], data_label_flow_2['data']):
        temp = []
        # for j in range(len(i[0])):
        temp.append(np.reshape(i, newshape=(2048 * 2, 1)))
        temp.append(np.reshape(f1, newshape=(2048 * 2, 1)))
        temp.append(np.reshape(f2, newshape=(2048 * 2, 1)))
        data.append(temp)
    return {'data': np.array(data), 'label': data_label_image['label']}


def reformat_flow(data_label_flow_1, data_label_flow_2):
    data = []
    for f1, f2 in zip(data_label_flow_1['data'], data_label_flow_2['data']):
        temp = []
        temp.append(f1)
        temp.append(f2)
        data.append(temp)
    return {'data': np.array(data), 'label': data_label_flow_1['label']}


def ws_flows(flow1_path, flow2_path, save_path1, save_path2, dim):
    # flow 1
    nr1 = Npyfilereader(flow1_path)
    nr1.validate(save_path1)
    # flow 2
    nr2 = Npyfilereader(flow2_path)
    nr2.validate(save_path2)

    video_num = len(nr1.npy_paths)
    for i in range(video_num):
        name1, contents1 = nr1.read_npys()
        name2, contents2 = nr2.read_npys()
        ws1 = Weightedsum(name1, contents1, save_path1)
        ws2 = Weightedsum(name2, contents2, save_path2)
        if dim == 0:
            ws1.mean_descriptor_gen()
            ws2.mean_descriptor_gen()
        else:
            trans_m = ws1.transformation_matrix_gen(dim, ws1.frame_features.shape[0])
            ws1.ws_descriptor_gen(dim, trans_matrix=trans_m)
            ws2.ws_descriptor_gen(dim, trans_matrix=trans_m)


def check_ws_existence(resNet_feature_save_path, resNet_ws_save_path, dim, flip=False):
    feature = []
    for (dirpath, dirnames, filenames) in os.walk(resNet_ws_save_path):
        feature += [f for f in filenames if f.endswith('.npy')]
    if len(feature) == 0:
        calWeightedSum.calculate_weightedsum(resNet_feature_save_path, resNet_ws_save_path, dim, flip)


def train_test_split_encode(tts, resNet_ws_save_path, dataset, split_num, encoder):
    # resNet image feature
    train_data_label_image, test_data_label_image = tts.train_test_split(resNet_ws_save_path, dataset, split_num)
    # normalize the data and encode labels
    train_data_label_image, test_data_label_image = norm_encode_data(train_data_label_image, test_data_label_image,
                                                                     encoder)
    return train_data_label_image, test_data_label_image


def main(resNet_save_path, resNet_ws_save_path, resNet_flip_ws_save_path,
         resNet_crop_save_path, resNet_crop_ws_save_path, resNet_crop_flip_ws_save_path,
         resNet_flow_save_path_1, resNet_flow_ws_save_path_1, resNet_flow_flip_ws_save_path_1,
         resNet_flow_crop_save_path_1, resNet_crop_flow_ws_save_path_1, resNet_crop_flip_flow_ws_save_path_1,
         resNet_flow_save_path_2, resNet_flow_ws_save_path_2, resNet_flow_flip_ws_save_path_2,
         resNet_flow_crop_save_path_2, resNet_crop_flow_ws_save_path_2, resNet_crop_flip_flow_ws_save_path_2,
         train_test_splits_save_path, dim, dataset='ucf'):
    # check the existence of ws features
    check_ws_existence(resNet_save_path, resNet_ws_save_path, dim)
    check_ws_existence(resNet_crop_save_path, resNet_crop_ws_save_path, dim)
    check_ws_existence(resNet_save_path, resNet_flip_ws_save_path, dim, flip=True)
    check_ws_existence(resNet_crop_save_path, resNet_crop_flip_ws_save_path, dim, flip=True)

    check_ws_existence(resNet_flow_save_path_1, resNet_flow_ws_save_path_1, dim)
    check_ws_existence(resNet_flow_crop_save_path_1, resNet_crop_flow_ws_save_path_1, dim)
    check_ws_existence(resNet_flow_save_path_1, resNet_flow_flip_ws_save_path_1, dim, flip=True)
    check_ws_existence(resNet_flow_crop_save_path_1, resNet_crop_flip_flow_ws_save_path_1, dim, flip=True)

    check_ws_existence(resNet_flow_save_path_2, resNet_flow_ws_save_path_2, dim)
    check_ws_existence(resNet_flow_crop_save_path_2, resNet_crop_flow_ws_save_path_2, dim)
    check_ws_existence(resNet_flow_save_path_2, resNet_flow_flip_ws_save_path_2, dim, flip=True)
    check_ws_existence(resNet_flow_crop_save_path_2, resNet_crop_flip_flow_ws_save_path_2, dim, flip=True)

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
        # train_data_label_image, test_data_label_image = train_test_split_encode(tts, resNet_ws_save_path, dataset,
        #                                                                         i, encoder)
        # # resNet flow u feature
        # train_data_label_flow_1, test_data_label_flow_1 = train_test_split_encode(tts, ucf_resNet_flow_ws_save_path_1,
        #                                                                           dataset, i, encoder)
        # # resNet flow v feature
        # train_data_label_flow_2, test_data_label_flow_2 = train_test_split_encode(tts, ucf_resNet_flow_ws_save_path_2,
        #                                                                           dataset, i, encoder)
        # # combine the image and flow features together with different channel
        # train_data_label = reformat(train_data_label_image, train_data_label_flow_1, train_data_label_flow_2)
        # test_data_label = reformat(test_data_label_image, test_data_label_flow_1, test_data_label_flow_2)

        # resNet image feature
        train_data_label_image, test_data_label_image = train_test_split_encode(tts, resNet_crop_ws_save_path, dataset,
                                                                                i, encoder)
        # resNet flow u feature
        train_data_label_flow_1, test_data_label_flow_1 = train_test_split_encode(tts, resNet_crop_flow_ws_save_path_1,
                                                                                  dataset, i, encoder)
        # resNet flow v feature
        train_data_label_flow_2, test_data_label_flow_2 = train_test_split_encode(tts, resNet_crop_flow_ws_save_path_2,
                                                                                  dataset, i, encoder)
        # combine the image and flow features together with different channel
        train_crop_data_label = reformat(train_data_label_image, train_data_label_flow_1,
                                         train_data_label_flow_2)
        test_crop_data_label = reformat(test_data_label_image, test_data_label_flow_1, test_data_label_flow_2)

        # # resNet image feature
        # train_data_label_image, test_data_label_image = train_test_split_encode(tts, resNet_flip_ws_save_path, dataset,
        #                                                                         i, encoder)
        # # resNet flow u feature
        # train_data_label_flow_1, test_data_label_flow_1 = train_test_split_encode(tts, resNet_flow_flip_ws_save_path_1,
        #                                                                           dataset, i, encoder)
        # # resNet flow v feature
        # train_data_label_flow_2, test_data_label_flow_2 = train_test_split_encode(tts, resNet_flow_flip_ws_save_path_2,
        #                                                                           dataset, i, encoder)
        # # combine the image and flow features together with different channel
        # train_data_label_flip = reformat_reshaped(train_data_label_image, train_data_label_flow_1,
        #                                           train_data_label_flow_2)
        # test_data_label_flip = reformat_reshaped(test_data_label_image, test_data_label_flow_1, test_data_label_flow_2)
        #
        # # resNet image feature
        # train_data_label_image, test_data_label_image = train_test_split_encode(tts, resNet_crop_flip_ws_save_path,
        #                                                                         dataset, i, encoder)
        # # resNet flow u feature
        # train_data_label_flow_1, test_data_label_flow_1 = train_test_split_encode(tts,
        #                                                                           resNet_crop_flip_flow_ws_save_path_1,
        #                                                                           dataset, i, encoder)
        # # resNet flow v feature
        # train_data_label_flow_2, test_data_label_flow_2 = train_test_split_encode(tts,
        #                                                                           resNet_crop_flip_flow_ws_save_path_2,
        #                                                                           dataset, i, encoder)
        # # combine the image and flow features together with different channel
        # train_data_label_crop_flip = reformat_reshaped(train_data_label_image, train_data_label_flow_1,
        #                                                train_data_label_flow_2)
        # test_data_label_crop_flip = reformat_reshaped(test_data_label_image, test_data_label_flow_1,
        #                                               test_data_label_flow_2)

        # 1
        # train_data_label['data'] = np.vstack((train_data_label['data'], train_crop_data_label['data'],
        #                                       ))
        # train_data_label['label'] = np.hstack((train_data_label['label'], train_crop_data_label['label'],
        #                                        ))
        #
        # test_data_label['data'] = np.vstack((test_data_label['data'], test_crop_data_label['data'],
        #                                     ))
        # test_data_label['label'] = np.hstack((test_data_label['label'], test_crop_data_label['label'],
        #                                       ))

        # 2
        # train_data_label['data'] = np.vstack(
        #     (np.concatenate((train_data_label['data'], train_crop_data_label['data']), axis=2),
        #      np.concatenate((train_data_label_flip['data'], train_data_label_crop_flip['data']), axis=2))
        # )
        # train_data_label['label'] = np.hstack((train_data_label['label'], train_data_label_flip['label']))
        #
        # test_data_label['data'] = np.vstack(
        #     (np.concatenate((test_data_label['data'], test_crop_data_label['data']), axis=2),
        #      np.concatenate((test_data_label_flip['data'], test_data_label_crop_flip['data']), axis=2))
        # )
        # test_data_label['label'] = np.hstack((test_data_label['label'], test_data_label_flip['label']))

        # # 3
        # train_data_label['data'] = np.vstack(
        #     (np.concatenate((train_data_label['data'], train_data_label_flip['data']), axis=-1),
        #      np.concatenate((train_crop_data_label['data'], train_data_label_crop_flip['data']), axis=-1))
        # )
        # train_data_label['label'] = np.hstack((train_data_label['label'], train_data_label_flip['label']))
        #
        # test_data_label['data'] = np.vstack(
        #     (np.concatenate((test_data_label['data'], test_data_label_flip['data']), axis=-1),
        #      np.concatenate((test_crop_data_label['data'], test_data_label_crop_flip['data']), axis=-1))
        # )
        # test_data_label['label'] = np.hstack((test_data_label['label'], test_data_label_flip['label']))

        # _train = np.unique(train_data_label['label'])
        # _test = np.unique(test_data_label['label'])
        #
        # print(_train)
        # print(_test)

        acc += onedconv_classifier.classify(train_crop_data_label['data'], train_crop_data_label['label'],
                                            test_crop_data_label['data'], test_crop_data_label['label'])
    print('accuracy for ', dataset, 'is', acc / (i + 1))


def remove_dirctories(directories):
    for d in directories:
        shutil.rmtree(d)
        os.makedirs(d)


def crop_main(resNet_crop_save_path_v1, resNet_flow_crop_save_path_1_v1, resNet_flow_crop_save_path_2_v1,
              resNet_crop_ws_save_path_v1, resNet_crop_flow_ws_save_path_1_v1,
              resNet_crop_flow_ws_save_path_2_v1,
              resNet_crop_save_path_v2, resNet_flow_crop_save_path_1_v2, resNet_flow_crop_save_path_2_v2,
              resNet_crop_ws_save_path_v2, resNet_crop_flow_ws_save_path_1_v2,
              resNet_crop_flow_ws_save_path_2_v2,
              train_test_splits_save_path, dim, dataset='ucf'):
    # check the existence of ws features
    check_ws_existence(resNet_crop_save_path_v1, resNet_crop_ws_save_path_v1, dim)
    check_ws_existence(resNet_flow_crop_save_path_1_v1, resNet_crop_flow_ws_save_path_1_v1, dim)
    check_ws_existence(resNet_flow_crop_save_path_2_v1, resNet_crop_flow_ws_save_path_2_v1, dim)

    check_ws_existence(resNet_crop_save_path_v2, resNet_crop_ws_save_path_v2, dim)
    check_ws_existence(resNet_flow_crop_save_path_1_v2, resNet_crop_flow_ws_save_path_1_v2, dim)
    check_ws_existence(resNet_flow_crop_save_path_2_v2, resNet_crop_flow_ws_save_path_2_v2, dim)

    if dataset == 'hmdb':
        tts = TrainTestSampleGen(ucf_path='', hmdb_path=train_test_splits_save_path)
    else:
        tts = TrainTestSampleGen(ucf_path=train_test_splits_save_path, hmdb_path='')

    acc = 0
    encoder = preprocessing.LabelEncoder()
    for i in range(1):
        # resNet image feature
        train_data_image_v1, test_data_image_v1 = train_test_split_encode(tts, resNet_crop_ws_save_path_v1, dataset,
                                                                          i, encoder)
        # resNet flow u feature
        train_data_flow1_v1, test_data_flow1_v1 = train_test_split_encode(tts, resNet_crop_flow_ws_save_path_1_v1,
                                                                          dataset, i, encoder)
        # resNet flow v feature
        train_data_flow2_v1, test_data_flow2_v1 = train_test_split_encode(tts, resNet_crop_flow_ws_save_path_2_v1,
                                                                          dataset, i, encoder)
        # combine the image and flow features together with different channel
        train_v1 = reformat(train_data_image_v1, train_data_flow1_v1,
                            train_data_flow2_v1)
        test_v1 = reformat(test_data_image_v1, test_data_flow1_v1, test_data_flow2_v1)
        #
        # train_data_image_v2, test_data_image_v2 = train_test_split_encode(tts, resNet_crop_ws_save_path_v2, dataset,
        #                                                                   i, encoder)
        # # resNet flow u feature
        # train_data_flow1_v2, test_data_flow1_v2 = train_test_split_encode(tts, resNet_crop_flow_ws_save_path_1_v2,
        #                                                                   dataset, i, encoder)
        # # resNet flow v feature
        # train_data_flow2_v2, test_data_flow2_v2 = train_test_split_encode(tts, resNet_crop_flow_ws_save_path_2_v2,
        #                                                                   dataset, i, encoder)
        # # combine the image and flow features together with different channel
        # train_v2 = reformat(train_data_image_v2, train_data_flow1_v2,
        #                              train_data_flow2_v2)
        # test_v2 = reformat(test_data_image_v2, test_data_flow1_v2, test_data_flow2_v2)

        # 1
        # train = {'data': np.vstack((train_v1['data'], train_v2['data'])),
        #          'label': np.hstack((train_v1['label'], train_v2['label']))}
        #
        # test = {'data': np.vstack((test_v1['data'], test_v2['data'])),
        #         'label': np.hstack((test_v1['label'], test_v2['label']))}

        acc += onedconv_classifier.classify(train_v1['data'], train_v1['label'],
                                            test_v1['data'], test_v1['label'])
    print('accuracy for ', dataset, 'is', acc / (i + 1))


if __name__ == '__main__':
    ucf_resNet_flip_ws_save_path = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flip_ws"
    ucf_resNet_flip_flow_ws_save_path_1 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow_flip_ws_u"
    ucf_resNet_flip_flow_ws_save_path_2 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow_flip_ws_v"

    ucf_resNet_crop_flip_ws_save_path = "/home/boy2/UCF101/ucf101_dataset/features/resNet_crop_flip_ws"
    ucf_resNet_crop_flip_flow_ws_save_path_1 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow_crop_flip_ws_u"
    ucf_resNet_crop_flip_flow_ws_save_path_2 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow_crop_flip_ws_v"

    ucf_resNet_ws_save_path = "/home/boy2/UCF101/ucf101_dataset/features/resNet_ws"
    ucf_resNet_flow_ws_save_path_1 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow_ws_u"
    ucf_resNet_flow_ws_save_path_2 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow_ws_v"

    ucf_resNet_crop_ws_save_path_v2 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_crop_ws_v2"
    ucf_resNet_crop_flow_ws_save_path_1_v2 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow_crop_ws_u_v2"
    ucf_resNet_crop_flow_ws_save_path_2_v2 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow_crop_ws_v_v2"

    ucf_resNet_flow_crop_save_path_1_v2 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop_v2/u"
    ucf_resNet_flow_crop_save_path_2_v2 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop_v2/v"
    ucf_resNet_crop_save_path_v2 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_crop_v2"

    ucf_resNet_crop_ws_save_path_v1 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_crop_ws_v1"
    ucf_resNet_crop_flow_ws_save_path_1_v1 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow_crop_ws_u_v1"
    ucf_resNet_crop_flow_ws_save_path_2_v1 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow_crop_ws_v_v1"

    ucf_resNet_flow_crop_save_path_1_v1 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop/u"
    ucf_resNet_flow_crop_save_path_2_v1 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop/v"
    ucf_resNet_crop_save_path_v1 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_crop"

    ucf_resNet_flow_save_path_1 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow/u"
    ucf_resNet_flow_save_path_2 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow/v"
    ucf_resNet_save_path = "/home/boy2/UCF101/ucf101_dataset/features/resNet"

    ucf_train_test_splits_save_path = "/home/boy2/UCF101/ucf101_dataset/features/testTrainSplits"

    remove_dirctories([ucf_resNet_crop_ws_save_path_v1, ucf_resNet_crop_flow_ws_save_path_1_v1,
                       ucf_resNet_crop_flow_ws_save_path_2_v1,
                       ucf_resNet_crop_ws_save_path_v2, ucf_resNet_crop_flow_ws_save_path_1_v2,
                       ucf_resNet_crop_flow_ws_save_path_2_v2])

    # main(ucf_resNet_save_path, ucf_resNet_ws_save_path, ucf_resNet_flip_ws_save_path,
    #      ucf_resNet_crop_save_path, ucf_resNet_crop_ws_save_path, ucf_resNet_crop_flip_ws_save_path,
    #      ucf_resNet_flow_save_path_1, ucf_resNet_flow_ws_save_path_1, ucf_resNet_flip_flow_ws_save_path_1,
    #      ucf_resNet_flow_crop_save_path_1, ucf_resNet_crop_flow_ws_save_path_1,
    #      ucf_resNet_crop_flip_flow_ws_save_path_1,
    #      ucf_resNet_flow_save_path_2, ucf_resNet_flow_ws_save_path_2, ucf_resNet_flip_flow_ws_save_path_2,
    #      ucf_resNet_flow_crop_save_path_2, ucf_resNet_crop_flow_ws_save_path_2,
    #      ucf_resNet_crop_flip_flow_ws_save_path_2,
    #      ucf_train_test_splits_save_path, 2)

    crop_main(ucf_resNet_crop_save_path_v1, ucf_resNet_flow_crop_save_path_1_v1, ucf_resNet_flow_crop_save_path_2_v1,
              ucf_resNet_crop_ws_save_path_v1, ucf_resNet_crop_flow_ws_save_path_1_v1,
              ucf_resNet_crop_flow_ws_save_path_2_v1,
              ucf_resNet_crop_save_path_v2, ucf_resNet_flow_crop_save_path_1_v2, ucf_resNet_flow_crop_save_path_2_v2,
              ucf_resNet_crop_ws_save_path_v2, ucf_resNet_crop_flow_ws_save_path_1_v2,
              ucf_resNet_crop_flow_ws_save_path_2_v2,
              ucf_train_test_splits_save_path, 2)
