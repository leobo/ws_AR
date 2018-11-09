import numpy as np
from sklearn import preprocessing

from npyFileReader import Npyfilereader
from onedConv_time import convTime
from trainTestSamplesGen import TrainTestSampleGen
from weighted_sum.videoDescriptorWeightedSum import Weightedsum
import test


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


def main(image_path, flow_path_u, flow_path_v, train_test_splits_save_path, dataset='ucf'):
    if dataset == 'hmdb':
        tts = TrainTestSampleGen(ucf_path='', hmdb_path=train_test_splits_save_path)
    else:
        tts = TrainTestSampleGen(ucf_path=train_test_splits_save_path, hmdb_path='')

    acc = 0
    for i in range(1):
        acc = test.classify(tts.ucf_train_data_label[i], tts.ucf_test_data_label[i], image_path, flow_path_u,
                                flow_path_v)
    print('accuracy for ', dataset, 'is', acc / (i + 1))


if __name__ == '__main__':
    ucf_resNet_flow_save_path_1 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow_crop/u"
    ucf_resNet_flow_save_path_2 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow_crop/v"
    ucf_resNet_save_path = "/home/boy2/UCF101/ucf101_dataset/features/resNet_crop"
    ucf_train_test_splits_save_path = "/home/boy2/UCF101/ucf101_dataset/features/testTrainSplits"
    main(ucf_resNet_save_path, ucf_resNet_flow_save_path_1, ucf_resNet_flow_save_path_2,
         ucf_train_test_splits_save_path)
