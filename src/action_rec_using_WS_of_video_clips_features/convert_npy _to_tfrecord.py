import tensorflow as tf
import numpy as np

from npyFileReader import Npyfilereader


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 141994 for train
def convert(npy_path, tfrecord_path, label_dict):
    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    reader = Npyfilereader(npy_path)
    while len(reader.npy_paths) != 0:
        _, contents = reader.read_npys()
        if _ != "path_label_list.npy":
            contents = contents.item()
            for key, value in contents.items():
                feature = {'label': _int64_feature(int(label_dict[key])),
                           'feature': _bytes_feature(value.astype(np.float32).tobytes())}
                # print(feature)
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())
    writer.close()


def read_labels(path):
    label_dict = {}
    with open(path, 'r') as reader:
        contents = reader.readlines()
        for con in contents:
            label_dict[con.split()[1]] = con.split()[0]
    return label_dict


if __name__ == '__main__':
    temp_train_store_path = "/home/boy2/UCF101/ucf101_dataset/features/important_resNet_crop_ws_train"
    temp_test_store_path = "/home/boy2/UCF101/ucf101_dataset/features/important_resNet_crop_ws_test"

    train_tfrecord = '/home/boy2/UCF101/ucf101_dataset/features/tfrecord_resNet_crop_ws/train.tfrecords'
    test_tfrecord = '/home/boy2/UCF101/ucf101_dataset/features/tfrecord_resNet_crop_ws/test.tfrecords'

    ucf_classes = "/home/boy2/UCF101/ucf101_dataset/features/testTrainSplits/classInd.txt"

    label_dict = read_labels(ucf_classes)

    convert(temp_train_store_path, train_tfrecord, label_dict)
    convert(temp_test_store_path, test_tfrecord, label_dict)
