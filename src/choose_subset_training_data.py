from trainTestSamplesGen import TrainTestSampleGen
from sklearn import preprocessing
import os
import random
import scipy.io as sio
from pathlib import Path
dataset = 'ucf'
ucf_train_test_splits_save_path = "/home/boy2/ucf101/ucf101_dataset/ucfTrainTestlist"
num_train_sample_per_label = 100
data_split = 0
num_sample_per_class = 80


if __name__ == '__main__':
    # generate the smaller train set for WS
    if dataset == 'hmdb':
        tts = TrainTestSampleGen(ucf_path='', hmdb_path=ucf_train_test_splits_save_path)
    else:
        tts = TrainTestSampleGen(ucf_path=ucf_train_test_splits_save_path, hmdb_path='')

    selected_data = dict()
    data_names, labels = tts.train_data_label[data_split]['data'], tts.train_data_label[data_split]['label']
    encoder = preprocessing.LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    for n, l, e in zip(data_names, labels, labels_encoded):
        if e in selected_data:
            selected_data[e] += [(n, l)]
        else:
            selected_data[e] = [(n, l)]

    store_path = os.path.join(ucf_train_test_splits_save_path, "trainlist" + str(num_train_sample_per_label) + ".txt")
    my_file = Path(store_path)
    if my_file.is_file():
        os.remove(store_path)

    name_list = list()
    for k in selected_data.keys():
        random.shuffle(selected_data[k])
        selected_pair = selected_data[k][:num_sample_per_class]
        for p in selected_pair:
            name, label = p
            name_list.append(name)
            line = label + "/" + name + '.avi' + ' ' + str(k) + '\n'
            with open(store_path, 'a') as f:
                f.write(line)

    # # generate the same train set for st-resnet
    # train_set_mat = sio.loadmat("/home/boy2/st-resnet/data/ucf101/ucf101_split1imdb.mat")
    # names = train_set_mat['images']['name'][0][0][0]
    # set = train_set_mat['images']['set'][0][0][0]
    # label = train_set_mat['images']['label'][0][0][0]
    # labels = train_set_mat['images']['labels'][0][0]
    # nFrames = train_set_mat['images']['nFrames'][0][0][0]
    # flowScales = train_set_mat['images']['flowScales'][0][0][0]
    #
    # keep_indx = list()
    # for i in range(13320):
    #     if names[i][0].split('.')[0] in name_list:
    #         keep_indx.append(i)
    # train_set_mat['images']['name'][0][0][0] = train_set_mat['images']['name'][0][0][0][keep_indx]
    # print()
