import os
import shutil

import numpy as np
from sklearn import preprocessing

from classifier import mulNet_onedconv_classifier
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
            temp_norm[:, :, i] = preprocessing.normalize(temp[:, :, i], axis=0)
    else:
        temp_norm = preprocessing.normalize(temp, axis=0)
    # optional: normalize each sample data independently
    # temp_norm = preprocessing.normalize(temp_norm, axis=1)
    train_data_label_image['data'] = temp_norm[:len_train]
    test_data_label_image['data'] = temp_norm[len_train:]
    train_data_label_image['label'] = encoder.fit_transform(train_data_label_image['label'])
    test_data_label_image['label'] = encoder.fit_transform(test_data_label_image['label'])
    return train_data_label_image, test_data_label_image


def reformat(data_label_image, data_label_flow):
    data = []
    for i, f1 in zip(data_label_image['data'], data_label_flow['data']):
        temp = []
        # for j in range(len(i[0])):
        temp.append(i)
        temp.append(f1)
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


def check_ws_existence(resNet_feature_save_path, resNet_ws_save_path, dim):
    feature = []
    for (dirpath, dirnames, filenames) in os.walk(resNet_ws_save_path):
        feature += [f for f in filenames if f.endswith('.npy')]
    if len(feature) == 0:
        calWeightedSum.calculate_swap_weightedsum(resNet_feature_save_path, resNet_ws_save_path, dim)


def train_test_split_encode(tts, resNet_ws_save_path, dataset, split_num, encoder):
    # resNet image feature
    train_data_label_image, test_data_label_image = tts.train_test_split(resNet_ws_save_path, dataset, split_num)
    # normalize the data and encode labels
    train_data_label_image, test_data_label_image = norm_encode_data(train_data_label_image, test_data_label_image,
                                                                     encoder)
    return train_data_label_image, test_data_label_image


def main(ucf_mulNet_rgb_origin_path, ucf_mulNet_ws_rgb_origin_path,
         ucf_mulNet_rgb_flip_path, ucf_mulNet_ws_rgb_flip_path,
         ucf_mulNet_flow_origin_path,ucf_mulNet_ws_flow_origin_path,
         ucf_mulNet_flow_flip_path,ucf_mulNet_ws_flow_flip_path,
         train_test_splits_save_path, dim, dataset='ucf'):
    # check the existence of ws features
    #check_ws_existence(ucf_mulNet_rgb_origin_path, ucf_mulNet_ws_rgb_origin_path, dim)
    #check_ws_existence(ucf_mulNet_rgb_flip_path, ucf_mulNet_ws_rgb_flip_path, dim)

    check_ws_existence(ucf_mulNet_flow_origin_path, ucf_mulNet_ws_flow_origin_path, dim)
    #check_ws_existence(ucf_mulNet_flow_flip_path, ucf_mulNet_ws_flow_flip_path, dim)


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
        train_data_origin_label_image, test_data_origin_label_image = train_test_split_encode(tts,
                                                                                              ucf_mulNet_ws_rgb_origin_path,
                                                                                              dataset, i, encoder)
        train_data_flip_label_image, test_data_flip_label_image = train_test_split_encode(tts,
                                                                                          ucf_mulNet_ws_rgb_flip_path,
                                                                                          dataset, i, encoder)
        # resNet flow orginal feature
        train_data_origin_label_flow, test_data_origin_label_flow = train_test_split_encode(tts,
                                                                                            ucf_mulNet_ws_flow_origin_path,
                                                                                            dataset, i, encoder)

        # resNet flow flipped feature
        train_data_flip_label_flow, test_data_flip_label_flow = train_test_split_encode(tts, ucf_mulNet_ws_flow_flip_path,
                                                                                        dataset, i, encoder)

        # combine the image and flow features together with different channel (origin version)
        train_data_label = reformat(train_data_origin_label_image, train_data_origin_label_flow)
        test_data_label = reformat(test_data_origin_label_image, test_data_origin_label_flow)

        #combine the image and flow feathres together with different channel (flipped version)
        train_flip_data_label = reformat(train_data_flip_label_image, train_data_flip_label_flow)
        test_flip_data_label = reformat(test_data_flip_label_image, test_data_flip_label_flow)


        train_data_label['data'] = np.vstack((train_data_label['data'], train_flip_data_label['data']))
        train_data_label['label'] = np.hstack((train_data_label['label'], train_flip_data_label['label']))

        test_data_label['data'] = np.vstack((test_data_label['data'], test_flip_data_label['data']))
        test_data_label['label'] = np.hstack((test_data_label['label'], test_flip_data_label['label']))

        acc += mulNet_onedconv_classifier.classify(train_data_label['data'], train_data_label['label'],
                                            test_data_label['data'], test_data_label['label'])['accuracy']
    print('accuracy for ', dataset, 'is', acc / (i + 1))


def remove_dirctories(directories):
    for d in directories:
        shutil.rmtree(d)
        os.makedirs(d)


if __name__ == '__main__':
    #raw data directory
    ucf_mulNet_rgb_origin_path = "/home/boy2/UCF101/ucf101_dataset/features/mulNet/rgb/o"
    ucf_mulNet_rgb_flip_path = "/home/boy2/UCF101/ucf101_dataset/features/mulNet/rgb/r"
    ucf_mulNet_flow_origin_path = "/home/boy2/UCF101/ucf101_dataset/features/mulNet/flow/o"
    ucf_mulNet_flow_flip_path = "/home/boy2/UCF101/ucf101_dataset/features/mulNet/flow/r"
    #calculated data export path
    ucf_mulNet_ws_rgb_origin_path = "/home/boy2/UCF101/ucf101_dataset/features/mulNet_ws/rgb/o"
    ucf_mulNet_ws_rgb_flip_path = "/home/boy2/UCF101/ucf101_dataset/features/mulNet_ws/rgb/f"
    ucf_mulNet_ws_flow_origin_path = "/home/boy2/UCF101/ucf101_dataset/features/mulNet_ws/flow/o"
    ucf_mulNet_ws_flow_flip_path = "/home/boy2/UCF101/ucf101_dataset/features/mulNet_ws/flow/f"

    ucf_train_test_splits_save_path = "/home/boy2/UCF101/ucf101_dataset/features/mulNetTrainSplits"

    # remove_dirctories([ucf_resNet_ws_save_path, ucf_resNet_flow_ws_save_path_1, ucf_resNet_flow_ws_save_path_2,
    #                    ucf_resNet_crop_ws_save_path, ucf_resNet_crop_flow_ws_save_path_1,
    #                    ucf_resNet_crop_flow_ws_save_path_2])

    main(ucf_mulNet_rgb_origin_path, ucf_mulNet_ws_rgb_origin_path, ucf_mulNet_rgb_flip_path,
         ucf_mulNet_ws_rgb_flip_path, ucf_mulNet_flow_origin_path, ucf_mulNet_ws_flow_origin_path,
         ucf_mulNet_flow_flip_path, ucf_mulNet_ws_flow_flip_path, ucf_train_test_splits_save_path, 2)
