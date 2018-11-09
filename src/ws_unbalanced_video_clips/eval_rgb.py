import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import tensorflow as tf
from sklearn import preprocessing, svm

from trainTestSamplesGen import TrainTestSampleGen
from weighted_sum.videoDescriptorWeightedSum import Weightedsum
from ws_unbalanced_video_clips.class_rgb import classify

# from ws_unbalanced_video_clips.two_stream_classifier import classify
train_steps = 10
test_steps = 5
dims = 16
num_cha = 9
min_len_video = 30
max_len_video = 1000
num_train_data = 9537
num_test_data = 3783
# num_train_data = 10
# num_test_data = 10
# num_train = np.random.randint(0, 3783, size=10)
# num_test = np.random.randint(0, 3783, size=10)
num_trans_matrices = 1
num_samples_per_training_video = 8
num_samples_per_testing_video = 8


def fisher_vector(xx, gmm):
    """Computes the Fisher vector on a set of descriptors.
    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors
    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.
    Returns
    -------
    fv: array_like, shape (K + 2 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.
    Reference
    ---------
    J. Krapac, J. Verbeek, F. Jurie.  Modeling Spatial Layout with Fisher
    Vectors for Image Categorization.  In ICCV, 2011.
    http://hal.inria.fr/docs/00/61/94/03/PDF/final.r1.pdf
    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
            - Q_xx_2
            - Q_sum * gmm.means_ ** 2
            + Q_sum * gmm.converged_
            + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))


def norm_encode_data(train_data, test_data):
    len_train = len(train_data)
    # normalize the data in time direction
    temp = np.concatenate([train_data, test_data])
    if len(temp.shape) == 3:
        x, y, t = temp.shape
        temp_norm = np.zeros(shape=(x, y, t))
        for i in range(t):
            temp_norm[:, :, i] = preprocessing.normalize(temp[:, :, i], axis=0, norm='max')
            # temp_norm[:, :, i] = preprocessing.normalize(temp_norm[:, :, i], axis=1, norm='max')
            # temp_norm[:, :, i] = preprocessing.scale(temp[:, :, i], axis=0)
    else:
        temp_norm = preprocessing.normalize(temp, axis=1, norm='max')
    # optional: normalize each sample data independently
    # temp_norm = preprocessing.normalize(temp_norm, axis=1)
    train_data = temp_norm[:len_train]
    test_data = temp_norm[len_train:]
    return train_data, test_data


def find_max(train_data, test_data):
    # normalize the data in time direction
    temp = np.concatenate([train_data, test_data])
    m = []
    if len(temp.shape) == 3:
        x, y, t = temp.shape
        for i in range(t):
            m.append(np.max(temp[:, :, i], axis=0))
    else:
        temp_norm = preprocessing.normalize(temp, axis=1, norm='max')
    return np.max(m, axis=0)
    # return m


def scaler_encode_data(train_data, test_data):
    len_train = len(train_data)
    # normalize the data in time direction
    temp = np.concatenate([train_data, test_data])
    if len(temp.shape) == 3:
        x, y, t = temp.shape
        temp_norm = np.zeros(shape=(x, y, t))
        scaler = preprocessing.StandardScaler()
        for i in range(t):
            scaler.fit(temp_norm[:, :, i])
        for i in range(t):
            temp_norm[:, :, i] = scaler.transform(temp[:, :, i])

    else:
        temp_norm = preprocessing.normalize(temp, axis=1, norm='max')
    # optional: normalize each sample data independently
    # temp_norm = preprocessing.normalize(temp_norm, axis=1)
    train_data = temp_norm[:len_train]
    test_data = temp_norm[len_train:]
    return train_data, test_data


def norm_encode_data_v2(train_data_1, test_data_1, train_data_2, test_data_2):
    len_train = len(train_data_1)
    len_test = len(test_data_1)
    # normalize the data in time direction
    temp = np.concatenate([train_data_1, train_data_2, test_data_1, test_data_2])
    if len(temp.shape) == 3:
        x, y, t = temp.shape
        temp_norm = np.zeros(shape=(x, y, t))
        for i in range(t):
            temp_norm[:, :, i] = preprocessing.normalize(temp[:, :, i], axis=0, norm='max')
            # temp_norm[:, :, i] = preprocessing.normalize(temp_norm[:, :, i], axis=1, norm='max')
            # temp_norm[:, :, i] = preprocessing.scale(temp[:, :, i], axis=0)
    else:
        temp_norm = preprocessing.normalize(temp, axis=1, norm='max')
    # optional: normalize each sample data independently
    # temp_norm = preprocessing.normalize(temp_norm, axis=1)
    train_data_1 = temp_norm[:len_train]
    train_data_2 = temp_norm[len_train:2 * len_train]
    test_data_1 = temp_norm[2 * len_train:2 * len_train + len_test]
    test_data_2 = temp_norm[2 * len_train + len_test:]
    return train_data_1, test_data_1, train_data_2, test_data_2


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def tf_record_writer_eval_v2(path, name, rgb1, flow1, label1, rgb2, flow2, label2):
    if not os.path.exists(path):
        os.mkdir(path)
    writer = tf.python_io.TFRecordWriter(os.path.join(path, name))
    for _rgb1, _flow1, _label1, _rgb2, _flow2, _label2 in zip(rgb1, flow1, label1, rgb2, flow2, label2):
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
    writer.close()


def tf_record_writer_ch(path, name, rgb, flow, label, dims, num_cha):
    if not os.path.exists(path):
        os.mkdir(path)
    writer = tf.python_io.TFRecordWriter(os.path.join(path, name))
    for _rgb, _flow, _label in zip(rgb, flow, label):
        for i in range(int(dims / num_cha)):
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


def tf_record_writer(path, name, rgb, label):
    if not os.path.exists(path):
        os.mkdir(path)
    writer = tf.python_io.TFRecordWriter(os.path.join(path, name))
    for _rgb, _label in zip(rgb, label):
        feature = {'rgb': _bytes_feature(_rgb.astype(np.float32).tobytes()),
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


def input_gen_all(rgb_feature_path, selected_frames_path, video_names, video_labels,
                  seed, steps=5, dim=2, train=False):
    video_names = np.array(video_names)
    video_labels = video_labels[:num_train_data]
    video_names = video_names[:num_train_data]

    # video_labels = video_labels[num_train]
    # video_names = video_names[num_train]
    # 1776 the max len video
    rgb_path = [os.path.join(rgb_feature_path, n + '.npy') for n in video_names]
    selected_frames = [os.path.join(selected_frames_path, n + '.mat') for n in video_names]
    all_rgb = []
    all_label = []

    for rp, sf, name, l in zip(rgb_path, selected_frames, video_names, video_labels):
        rgb = np.load(rp)

        if len(seed.shape) == 1:
            tran_m_rgb = Weightedsum('tran_m', [], None).transformation_matrix_gen_norm(dims, len(rgb), seed)
            tran_m_rgb = preprocessing.normalize(tran_m_rgb, axis=1, norm='l2')
        else:
            tran_m_rgb = seed

        _rgb = Weightedsum(name, rgb[:], None).ws_descriptor_gen(dim, False, tran_m_rgb)

        all_rgb.append(_rgb)
        all_label.append(l)

    return all_rgb, all_label


def crop_main(rgb_frame_feature_list, ucf_resNet_train_path,
              ucf_resNet_test_path, train_test_splits_save_path, dims, selected_frames_path, seeds,
              dataset='ucf'):
    if dataset == 'hmdb':
        tts = TrainTestSampleGen(ucf_path='', hmdb_path=train_test_splits_save_path)
    else:
        tts = TrainTestSampleGen(ucf_path=train_test_splits_save_path, hmdb_path='')

    encoder = preprocessing.LabelEncoder()
    for i in range(1):
        j = 0
        max_rgb = []
        for m in seeds:
            for rgb in rgb_frame_feature_list:
                t_rgb, t_label = input_gen_all(rgb, selected_frames_path,
                                               tts.train_data_label[i]['data'],
                                               encoder.fit_transform(tts.train_data_label[i]['label']),
                                               m, steps=train_steps, dim=dims)
                e_rgb, e_label = input_gen_all(rgb, selected_frames_path,
                                               tts.test_data_label[i]['data'],
                                               encoder.fit_transform(tts.test_data_label[i]['label']),
                                               m, steps=train_steps, dim=dims)
                max_rgb.append(find_max(t_rgb, e_rgb))
                tf_record_writer(ucf_resNet_train_path, "rgb_labels_" + str(j) + ".tfrecord", t_rgb,
                                 t_label)
                tf_record_writer(ucf_resNet_test_path, "rgb_labels_" + str(j) + ".tfrecord", e_rgb,
                                 e_label)
                print("Version", j, "train-test data generation done")
                j += 1

        max_rgb = np.max(max_rgb, axis=0)
        return max_rgb.astype(dtype=np.float32), encoder.fit_transform(tts.train_data_label[i]['label'])


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
    # ucf_resNet_flow_crop_save_path_1_v1 = "/home/boy2/UCF101/ucf101_dataset/features/mulNet_feature/mulNet/flow/o"
    ucf_resNet_flow_crop_save_path_2_v1 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop/v"
    ucf_resNet_crop_save_path_v1 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_crop"
    # ucf_resNet_crop_save_path_v1 = "/home/boy2/UCF101/ucf101_dataset/features/mulNet_feature/mulNet/rgb/o"

    ucf_resNet_flow_save_path_1 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow/u"
    ucf_resNet_flow_save_path_2 = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow/v"
    ucf_resNet_save_path = "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet"

    featureStorePath = ["/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_resize",
                        "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_resize/u",
                        "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_resize/v"]
    featureStorePathFlip = ["/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_resize_flip",
                            "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_resize_flip/u",
                            "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_resize_flip/v"]

    ucf_resNet_train_path_v1 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_train_v1"
    ucf_resNet_test_path_v1 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_test_v1"

    ucf_resNet_train_path_v2 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_train_v2"
    ucf_resNet_test_path_v2 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_test_v2"

    ucf_resNet_train_path_v3 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_train_v3"
    ucf_resNet_test_path_v3 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_test_v3"

    ucf_resNet_train_path_v4 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_train_v4"
    ucf_resNet_test_path_v4 = "/home/boy2/UCF101/ucf101_dataset/features/unbalanced_video_clips_test_v4"

    ucf_train_test_splits_save_path = "/home/boy2/UCF101/ucf101_dataset/features/testTrainSplits"

    selected_frames_path = '/home/boy2/UCF101/ucf101_dataset/frame_features/seq_number'

    lto = ["/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_top_o/rgb",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_top_o/u",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_top_o/v"]
    ltf = ["/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_top_f/rgb",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_top_f/u",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_top_f/v"]
    lbo = ["/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_bottom_o/rgb",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_bottom_o/u",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_bottom_o/v"]
    lbf = ["/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_bottom_f/rgb",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_bottom_f/u",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/left_bottom_f/v"]
    rto = ["/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_top_o/rgb",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_top_o/u",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_top_o/v"]
    rtf = ["/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_top_f/rgb",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_top_f/u",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_top_f/v"]
    rbo = ["/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_bottom_o/rgb",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_bottom_o/u",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_bottom_o/v"]
    rbf = ["/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_bottom_f/rgb",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_bottom_f/u",
           "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/right_bottom_f/v"]
    co = ["/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/center_o_meansub/rgb",
          "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/center_o_meansub/u",
          "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/center_o_meansub/v"]
    cf = ["/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/center_f/rgb",
          "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/center_f/u",
          "/home/boy2/UCF101/ucf101_dataset/stricted_cropped_frame_feature/center_f/v"]

    rgb = [e[0] for e in [lto, ltf, lbo, lbf, rto, rtf, rbo, rbf, co, cf]]
    u = [e[1] for e in [lto, ltf, lbo, lbf, rto, rtf, rbo, rbf, co, cf]]
    v = [e[2] for e in [lto, ltf, lbo, lbf, rto, rtf, rbo, rbf, co, cf]]

    # remove_dirctories([ucf_resNet_crop_ws_save_path_v1, ucf_resNet_crop_flow_ws_save_path_1_v1,
    #                    ucf_resNet_crop_flow_ws_save_path_2_v1,
    #                    ucf_resNet_crop_ws_save_path_v2, ucf_resNet_crop_flow_ws_save_path_1_v2,
    #                    ucf_resNet_crop_flow_ws_save_path_2_v2])

    best_acc = 0
    acc_list = []
    seed_list = []
    loc_list = []
    tran_m_list = []
    tran_m_list_norm = []
    # for i in range(int(num_trans_matrices)):
    #     seed = np.random.randint(1, 100)
    #     # seed_list.append(seed)
    #     print("The seed for trans_matrix is:", seed)
    #     tran_m = Weightedsum('tran_m', [], None).transformation_matrix_gen(dims, 1776, seed)
    #     tran_m_list.append(tran_m)
    #     tran_m_list_norm.append(preprocessing.normalize(tran_m, axis=1, norm='l2'))

    # s = np.random.randint(0, dims, size=[num_trans_matrices-int(num_trans_matrices/2), int(np.floor((dims-1)/2))])
    loc = np.random.uniform(-1, 1, size=[int(num_trans_matrices), dims - 1])
    loc.sort()
    # for i in range(num_trans_matrices-int(num_trans_matrices/2)):
    #     seed_list.append(np.array([s[i], loc[i]]))

    for s in loc:
        tran_m_list_norm.append(s)

    # tran_m_list_norm = loc

    # tran_m_list_norm = np.array(tran_m_list_norm)
    np.save('/home/boy2/UCF101/ucf101_dataset/frame_features/seed.npy', seed_list)
    # np.save('/home/boy2/UCF101/ucf101_dataset/frame_features/tras_m_1.npy', tran_m_list)
    # np.save('/home/boy2/UCF101/ucf101_dataset/frame_features/tras_m_1_norm.npy', tran_m_list_norm)

    # max_value, test_gt = crop_main(
    #     rgb[:],
    #     ucf_resNet_train_path_v1, ucf_resNet_test_path_v1,
    #     ucf_train_test_splits_save_path, dims, selected_frames_path, tran_m_list_norm)
    # np.save(os.path.join(ucf_resNet_train_path_v1, "max_value.npy"), max_value)
    accuracy = classify(
        # os.path.join(ucf_resNet_train_path_v1, "rgb_flow_labels.tfrecord"),
        [os.path.join(ucf_resNet_train_path_v1, "rgb_labels_" + str(i) + ".tfrecord") for i in
         range(num_samples_per_training_video * num_trans_matrices)],
        [os.path.join(ucf_resNet_test_path_v1, "rgb_labels_" + str(i) + ".tfrecord") for i in
         range(num_samples_per_testing_video * num_trans_matrices)],
        num_train_data * num_samples_per_training_video * num_trans_matrices,
        num_test_data * num_samples_per_testing_video * num_trans_matrices,
        num_samples_per_testing_video,
        num_trans_matrices,
        (1, 2048, dims), os.path.join(ucf_resNet_train_path_v1, "max_value.npy"))
    print("accuracy for the current experiment is", accuracy)
    acc_list.append(accuracy)
    if accuracy > best_acc:
        best_acc = accuracy
    print("Accuracy for each exp is:", acc_list)
    np.savetxt('/home/boy2/UCF101/ucf101_dataset/frame_features/result.txt', acc_list, delimiter=',')
