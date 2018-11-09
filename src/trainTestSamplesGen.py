import os

import numpy as np
from preprocessing.cropper import Cropper



class TrainTestSampleGen(object):

    def __init__(self, ucf_path, hmdb_path):
        if ucf_path != '':
            self.train_data_label, self.test_data_label = self.ucf_read_train_test_split(ucf_path)
        if hmdb_path != '':
            self.train_data_label, self.test_data_label = self.hmdb_read_train_test_split(hmdb_path)

    def ucf_read_train_test_split(self, path):
        """
        Read the name of train and test samples from the txt files which are stored under path, split them into train,
        test lists.
        :param path: The path of folder stores the train test split txt files
        :return: two lists of dict, train and test which contain the video name and labels for train and test.
        """
        # get the test train split txt file
        train = []
        test = []
        for (dirpath, dirnames, filenames) in os.walk(path):
            train += [os.path.join(path, file) for file in filenames if file.startswith('trainlist')]
            test += [os.path.join(path, file) for file in filenames if file.startswith('testlist')]
            train.sort()
            test.sort()

        # read test train data name and label from the txt file
        train_data_labels = []
        test_data_labels = []
        for tra, test in zip(train, test):
            with open(tra) as f:
                names_labels = f.readlines()
                data = [line.split(' ')[0].split('/')[-1].split('.')[0] for line in names_labels]
                label = [line.split(' ')[0].split('/')[0] for line in names_labels]
                train_data_labels.append({'data': data, 'label': label})
            with open(test) as f:
                names_labels = f.readlines()
                data = [line.split('/')[-1].split('.')[0] for line in names_labels]
                label = [line.split('/')[0] for line in names_labels]
                test_data_labels.append({'data': data, 'label': label})
        return train_data_labels, test_data_labels

    def train_test_split(self, path, dataset, idx, crop=False):
        """
        Read the data that names are given in self.ucf_train_data_labels and self.ucf_test_data_labels.
        Save the data and label into a dict. Each time this function is called, only read one train or test split.
        :param path: feature store path
        :param dataset: ucf or hmdb
        :return: the dicts with the format {'data': data, 'label': labels}
        """
        if dataset == 'ucf':
            train_data_label = self.ucf_train_data_label[idx].copy()
            test_data_label = self.ucf_test_data_label[idx].copy()
        else:
            train_data_label = self.hmdb_train_data_label[idx].copy()
            test_data_label = self.hmdb_test_data_label[idx].copy()

        if crop == True:
            train_data = [Cropper(np.load(os.path.join(path, name + '.npy')), (224, 224)).crop_image() for name in
                          train_data_label['data']]
            test_data = [Cropper(np.load(os.path.join(path, name + '.npy')), (224, 224)).crop_image() for name in
                          test_data_label['data']]
        else:
            train_data = [np.load(os.path.join(path, name + '.npy')) for name in train_data_label['data']]
            test_data = [np.load(os.path.join(path, name + '.npy')) for name in test_data_label['data']]

        train_data_label['data'] = train_data
        test_data_label['data'] = test_data
        return train_data_label, test_data_label

    def hmdb_read_train_test_split(self, path):
        """
        Read the name of train and test samples from the txt files which are stored under path, split them into train,
        test lists.
        :param path: The path of folder stores the train test split txt files
        :return: two lists of dict, train and test which contain the video name and labels for train and test.
        """
        # read file names according to the split number from 1 to 3
        action_splits = []
        for (dirpath, dirnames, filenames) in os.walk(path):
            for i in range(1, 4, 1):
                action_splits.append(
                    sorted([os.path.join(path, f) for f in filenames if f.split('.')[0].endswith(str(i))]))

        # fetch the data and labels for all 3 splits
        train_data_labels = []
        test_data_labels = []
        for s in action_splits:
            train_data = []
            train_label = []
            test_data = []
            test_label = []
            for a in s:
                with open(a) as f:
                    name_labels = f.readlines()
                    train = [line.split(' ')[0].split('.')[0] for line in name_labels if
                                   line.rstrip().split(' ')[-1] == '1']
                    train_data += train
                    train_label += [a.split('/')[-1][:a.split('/')[-1].index('test')-1] for i in range(len(train))]
                    test = [line.split(' ')[0].split('.')[0] for line in name_labels if
                                  line.rstrip().split(' ')[-1] == '2']
                    test_data += test
                    test_label += [a.split('/')[-1][:a.split('/')[-1].index('test')-1] for i in range(len(test))]
            train_data_labels.append({'data': train_data, 'label': np.array(train_label)})
            test_data_labels.append({'data': test_data, 'label': np.array(test_label)})
        return train_data_labels, test_data_labels

if __name__ == '__main__':
    ucf_path = "/home/boy2/UCF101/ucf101_dataset/features/testTrainSplits"
    hmdb_path = "/home/boy2/UCF101/hmdb51_dataset/features/testTrainSplits"
    tts = TrainTestSampleGen(ucf_path, hmdb_path)
    tts.hmdb_read_train_test_split(hmdb_path)
