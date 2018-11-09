import numpy as np
from sklearn import preprocessing

from action_rec_using_WS_of_video_clips_features.ond_d_conNet_model_without_estimator import classify
# from action_rec_using_WS_of_video_clips_features.one_d_conNet_model import classify, classify_tfrecord
from npyFileReader import Npyfilereader
from action_rec_using_WS_of_video_clips_features import video_clips_creation


def read_data_label(save_path):
    data = []
    labels = []
    reader = Npyfilereader(save_path)
    reader.npy_paths = reader.npy_paths[:100]
    while len(reader.npy_paths) != 0:
        _, contents = reader.read_npys()
        if _ != "path_label_list.npy":
            contents = contents.item()
            for key, value in contents.items():
                data.append(value)
                labels.append(key)
    return {'data': np.array(data), 'label': np.array(labels)}


def main(temp_train_store_path, temp_test_store_path):
    acc = 0

    train_list = read_data_label(temp_train_store_path)
    test_list = read_data_label(temp_test_store_path)

    encoder = preprocessing.LabelEncoder()
    train_list['label'] = encoder.fit_transform(train_list['label']) + 1
    test_list['label'] = encoder.fit_transform(test_list['label']) + 1

    _train = np.unique(train_list['label'])
    _test = np.unique(test_list['label'])

    print(_train)
    print(_test)

    acc += classify(train_list, test_list)
    print('accuracy for ucf', 'is', acc)


def main_tfrecords(train_records, test_records):
    num_train_samples = 141993
    num_test_samples = 55362
    num_test_samples_per_video = 128
    classify(train_records, test_records, num_train_samples, num_test_samples, num_test_samples_per_video)


if __name__ == '__main__':
    ucf_resNet_flow_crop_save_path_1 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow_crop/u"
    ucf_resNet_flow_crop_save_path_2 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow_crop/v"
    ucf_resNet_crop_save_path = "/home/boy2/UCF101/ucf101_dataset/features/resNet_crop"

    ucf_resNet_flow_save_path_1 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop/u"
    ucf_resNet_flow_save_path_2 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop/v"
    ucf_resNet_save_path = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_crop"

    ucf_train_test_splits_save_path = "/home/boy2/UCF101/ucf101_dataset/features/testTrainSplits"

    temp_train_store_path = "/home/boy2/UCF101/ucf101_dataset/features/important_resNet_crop_ws_train"
    temp_test_store_path = "/home/boy2/UCF101/ucf101_dataset/features/important_resNet_crop_ws_test"

    train_tfrecord = '/home/boy2/UCF101/ucf101_dataset/features/tfrecord_resNet_crop_ws/train.tfrecords'
    test_tfrecord = '/home/boy2/UCF101/ucf101_dataset/features/tfrecord_resNet_crop_ws/test.tfrecords'

    video_clips_creation.creat()

    # main(temp_train_store_path, temp_test_store_path)
    main_tfrecords(train_tfrecord, test_tfrecord)
