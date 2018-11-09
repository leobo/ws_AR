import os, sys

# nohup /home/boy2/anaconda3/envs/tensorflow/bin/python /home/boy2/ucf101/src/resNet-152/resNet152_for_ws.py > /dev/null 2>&1 &

os.environ["CUDA_VISIBLE_DEVICES"]="0"

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append('/home/boy2/anaconda3/envs/tensorflow/')

import re
import numpy as np
import tensorflow as tf
from sklearn import preprocessing

from models.research.slim.nets import resnet_v1
from trainTestSamplesGen import TrainTestSampleGen



slim = tf.contrib.slim

TESTLEN = 20

ws_image = ["/home/boy2/ucf101/ucf101_dataset/features/frame_ws_3"]
IMG_WIDTH = 342
IMG_HEIGHT = 256
seg_num = 25

INPUT_WIDTH = 224
INPUT_HEIGHT = 224

SIZE = [256, 224, 192, 168]

RGB_CHANNELS = 3
FLOW_CHANNELS = 10
BATCH_SIZE = 10

dropout_rate = 0.9
num_unique_classes = 101
LEARNING_RATE = 0.001

total_epoch = 120
train_epoch = 1

clip_length = 5
num_cha = 9
min_len_video = 30
max_len_video = 1000
num_train_data = 9537
num_test_data = 3783
num_trans_matrices = 1
num_frame_features = 1
num_sample_train = 1
num_sample_test = 25


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


def gen_subvideo_v2(rgb, u, v, steps):
    if len(rgb) == len(u) + 1:
        rgb = rgb[:-1]
    data_splits_indeces = np.array_split(np.arange(len(rgb)), steps)
    rgb_indeces = [np.random.choice(d) for d in data_splits_indeces]
    rgb_splits = [rgb[i] for i in rgb_indeces]
    flow_indeces = []
    for i in rgb_indeces:
        if i - 5 < 0:
            start = 0
            stop = 10
        elif i + 5 > len(rgb) - 1:
            stop = len(rgb)
            start = stop - 10
        else:
            start = i - 5
            stop = i + 5
        flow_indeces.append((start, stop))

    u_splits = [u[index[0]:index[1]] for index in flow_indeces]
    v_splits = [v[index[0]:index[1]] for index in flow_indeces]

    return rgb_splits, u_splits, v_splits


def process_image(image, channel):
    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=channel)
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])
    # # Normalize
    # image = image * 1.0 / 127.5 - 1.0
    return image


def read_images(imagepaths, flow, channel):
    # Convert to Tensor
    # imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    # Build a TF Queue, shuffle data
    image = tf.train.slice_input_producer([imagepaths], shuffle=False, num_epochs=None)

    rgb = process_image(image[0], channel)
    rgb = tf.expand_dims(rgb, 0)

    # cropped_boxes = cropped_boxes_gen()
    # # for rgb image
    # cropped_images = list()
    # # original frame
    # cropped_image = tf.image.crop_and_resize(rgb, boxes=cropped_boxes,
    #                                          box_ind=[0, 0, 0, 0], crop_size=[224, 224])
    # cropped_image = tf.concat([cropped_image, tf.image.resize_bilinear(
    #     tf.image.central_crop(rgb, np.random.choice(SIZE) * np.random.choice(SIZE) / IMG_HEIGHT / IMG_WIDTH),
    #     [224, 224])], axis=0)
    #
    # cropped_images.append(cropped_image)
    # # flipped frame
    # cropped_image = tf.image.crop_and_resize(tf.image.flip_left_right(rgb), boxes=cropped_boxes,
    #                                          box_ind=[0, 0, 0, 0], crop_size=[224, 224])
    # cropped_image = tf.concat([cropped_image, tf.image.resize_bilinear(
    #     tf.image.central_crop(tf.image.flip_left_right(rgb),
    #                           np.random.choice(SIZE) * np.random.choice(SIZE) / IMG_HEIGHT / IMG_WIDTH),
    #     [224, 224])], axis=0)
    #
    # rgb = tf.concat([cropped_images[0], cropped_image], axis=0)
    return rgb, image


def cropped_boxes_gen():
    return [[0, 0, np.random.choice(SIZE) / IMG_HEIGHT, np.random.choice(SIZE) / IMG_WIDTH],
            [1 - np.random.choice(SIZE) / IMG_HEIGHT, 0, 1, np.random.choice(SIZE) / IMG_WIDTH],
            [0, 1 - np.random.choice(SIZE) / IMG_WIDTH, np.random.choice(SIZE) / IMG_HEIGHT, 1],
            [1 - np.random.choice(SIZE) / IMG_HEIGHT, 1 - np.random.choice(SIZE) / IMG_WIDTH, 1, 1]]


def video_clips_selection_ind(video_name, rgb_fpath, u_fpath, v_fpath):
    rgb_path = os.path.join(rgb_fpath, video_name)
    u_path = os.path.join(u_fpath, video_name)
    v_path = os.path.join(v_fpath, video_name)

    rgb, u, v = list(), list(), list()

    rgb_frame_p = sort_numerically([os.path.join(rgb_path, f) for f in os.listdir(rgb_path) if f.endswith('jpg')])
    u_frame_p = sort_numerically([os.path.join(u_path, f) for f in os.listdir(u_path) if f.endswith('jpg')])
    v_frame_p = sort_numerically([os.path.join(v_path, f) for f in os.listdir(v_path) if f.endswith('jpg')])

    if len(rgb_frame_p) == len(u_frame_p) + 1:
        rgb_frame_p = rgb_frame_p[:-1]

    rgb += rgb_frame_p
    u += u_frame_p
    v += v_frame_p
    return rgb, u, v


def video_clips_selection(video_name, video_label, rgb_fpath, u_fpath, v_fpath):
    # video_name = video_name[:TESTLEN]
    # video_label = video_label[:TESTLEN]
    rgb_path = [os.path.join(rgb_fpath, n) for n in video_name]
    u_path = [os.path.join(u_fpath, n) for n in video_name]
    v_path = [os.path.join(v_fpath, n) for n in video_name]

    rgb, u, v, labels, current_name = list(), list(), list(), list(), list()
    num_frame_each_video = dict()

    for name, rgb_p, u_p, v_p, l in zip(video_name, rgb_path, u_path, v_path, video_label):
        rgb_frame_p = sort_numerically([os.path.join(rgb_p, f) for f in os.listdir(rgb_p) if f.endswith('jpg')])
        u_frame_p = sort_numerically([os.path.join(u_p, f) for f in os.listdir(u_p) if f.endswith('jpg')])
        v_frame_p = sort_numerically([os.path.join(v_p, f) for f in os.listdir(v_p) if f.endswith('jpg')])

        if len(rgb_frame_p) == len(u_frame_p) + 1:
            rgb_frame_p = rgb_frame_p[:-1]

        num_frame_each_video[name] = len(rgb_frame_p)
        rgb += rgb_frame_p
        u += u_frame_p
        v += v_frame_p
        labels += [l for i in range(len(rgb_frame_p))]
        current_name += [name for i in range(len(rgb_frame_p))]
    return rgb, u, v, labels, current_name, num_frame_each_video


def gen_frame_feature_resNet152(framePath, rgb_p, u_p, v_p, train_test_splits_save_path):
    """
    Read all video frames from framePath, then generate feature for each of them by resNet152. Store the features
    in featureStorePath.
    :return:
    """
    encoder = preprocessing.LabelEncoder()

    tts = TrainTestSampleGen(ucf_path=train_test_splits_save_path, hmdb_path='')
    data = tts.train_data_label[0]['data'] + tts.test_data_label[0]['data']
    labels = tts.train_data_label[0]['label'] + tts.test_data_label[0]['label']

    # check if exist for rgb
    ind = set()
    for i in range(len(data)):
        e = 1
        for p in rgb_p:
            if not os.path.isfile(os.path.join(p, data[i]+'.npy')):
                e = 0
        if e == 0:
            ind.add(i)

    # u
    for i in range(len(data)):
        e = 1
        for p in u_p:
            if not os.path.isfile(os.path.join(p, data[i]+'.npy')):
                e = 0
        if e == 0:
            ind.add(i)

    # v
    for i in range(len(data)):
        e = 1
        for p in v_p:
            if not os.path.isfile(os.path.join(p, data[i]+'.npy')):
                e = 0
        if e == 0:
            ind.add(i)

    for e in sorted(ind, reverse=True):
        print(e)
        data.pop(e)
        labels.pop(e)


    # labels = labels[len((os.listdir(u_p[0]))):]
    # [data.remove(f.split(".")[0]) for f in os.listdir(u_p[0]) if f.endswith('npy')]

    t_rgb_clips, t_u_clips, t_v_clips, t_labels, t_names, num_frame_each_video = video_clips_selection(
        data[:10],
        encoder.fit_transform(labels)[:10],
        framePath[0], framePath[1], framePath[2])

    # t_rgb_clips, t_u_clips, t_v_clips = video_clips_selection_ind(
    #     data[:],
    #     framePath[0], framePath[1], framePath[2])

    for clips, store_path in zip([t_rgb_clips, t_u_clips, t_v_clips], [rgb_p, u_p, v_p]):
        tf.reset_default_graph()
        # the input_layer return 1 iamge
        input_layer, name = read_images(clips, BATCH_SIZE, RGB_CHANNELS)

        cropped_boxes = tf.placeholder(tf.float32, shape=[4, 4], name='cb')
        # original frame
        cropped_image = tf.image.crop_and_resize(input_layer, boxes=cropped_boxes,
                                                 box_ind=[0, 0, 0, 0], crop_size=[224, 224])
        cropped_image = tf.concat([cropped_image, tf.image.resize_bilinear(
            tf.image.central_crop(input_layer, np.random.choice(SIZE) * np.random.choice(SIZE) / IMG_HEIGHT / IMG_WIDTH),
            [224, 224])], axis=0)

        # flipped frame
        # if u we need invert optical flow
        if store_path[0].split('/')[-1] is 'u':
            input_layer = -tf.image.flip_left_right(input_layer) + 255
        f_cropped_image = tf.image.crop_and_resize(input_layer, boxes=cropped_boxes,
                                                 box_ind=[0, 0, 0, 0], crop_size=[224, 224])
        f_cropped_image = tf.concat([f_cropped_image, tf.image.resize_bilinear(
            tf.image.central_crop(input_layer,
                                  np.random.choice(SIZE) * np.random.choice(SIZE) / IMG_HEIGHT / IMG_WIDTH),
            [224, 224])], axis=0)

        input_layer = tf.concat([cropped_image, f_cropped_image], axis=0)

        # # for test
        # with tf.Session() as sess:
        #     sess.run(tf.local_variables_initializer())
        #     sess.run(tf.global_variables_initializer())
        #     coord = tf.train.Coordinator()
        #     threads = tf.train.start_queue_runners(coord=coord)
        #     while not coord.should_stop():
        #         images, n, cb = sess.run([input_layer, name, cropped_boxes], feed_dict={'cb:0': cropped_boxes_gen()})
        #         print(n)
        #     coord.request_stop()
        #     coord.join(threads)

        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            resNet50, end_points = resnet_v1.resnet_v1_152(input_layer,
                                                            num_classes=None,
                                                            is_training=False,
                                                            global_pool=True,
                                                            output_stride=None,
                                                            spatial_squeeze=True,
                                                            reuse=tf.AUTO_REUSE,
                                                            scope='resnet_v1_152')
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, "/home/boy2/ucf101/src/resNet-152/resnet_v1_152.ckpt")
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for k in num_frame_each_video.keys():
                print("Generate resNet feature for", k)
                all_f = list()
                for i in range(int(np.ceil(num_frame_each_video[k] / 1))):
                    # read all video frames and generate frame features by resNet152
                    des = sess.run([end_points['resnet_v1_152/block4']], feed_dict={'cb:0': cropped_boxes_gen()})
                    all_f.append(np.reshape(des, newshape=[10, 7, 7, 2048]))
                all_f = np.array(all_f)
                for s, i in zip(store_path, range(len(store_path))):
                    if not os.path.exists(s):
                        os.makedirs(s)
                    np.save(os.path.join(s, k), all_f[:, i, :])
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    framePath = ["/home/boy2/ucf101/ucf101_dataset/frames/jpegs_256",
                 "/home/boy2/ucf101/ucf101_dataset/frames/tvl1_flow/u",
                 "/home/boy2/ucf101/ucf101_dataset/frames/tvl1_flow/v"]
    lto = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_left_top_o/rgb",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_left_top_o/u",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_left_top_o/v"]
    ltf = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_left_top_f/rgb",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_left_top_f/u",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_left_top_f/v"]
    lbo = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_left_bottom_o/rgb",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_left_bottom_o/u",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_left_bottom_o/v"]
    lbf = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_left_bottom_f/rgb",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_left_bottom_f/u",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_left_bottom_f/v"]
    rto = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_right_top_o/rgb",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_right_top_o/u",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_right_top_o/v"]
    rtf = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_right_top_f/rgb",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_right_top_f/u",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_right_top_f/v"]
    rbo = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_right_bottom_o/rgb",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_right_bottom_o/u",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_right_bottom_o/v"]
    rbf = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_right_bottom_f/rgb",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_right_bottom_f/u",
           "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_right_bottom_f/v"]
    co = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_center_o/rgb",
          "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_center_o/u",
          "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_center_o/v"]
    cf = ["/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_center_f/rgb",
          "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_center_f/u",
          "/home/boy2/ucf101/ucf101_dataset/stricted_cropped_frame_feature/random_center_f/v"]

    train_test_splits_save_path = "/home/boy2/ucf101/ucf101_dataset/ucfTrainTestlist"

    rgb = [e[0] for e in [lto, ltf, lbo, lbf, rto, rtf, rbo, rbf, co, cf]]
    u = [e[1] for e in [lto, ltf, lbo, lbf, rto, rtf, rbo, rbf, co, cf]]
    v = [e[2] for e in [lto, ltf, lbo, lbf, rto, rtf, rbo, rbf, co, cf]]

    gen_frame_feature_resNet152(framePath, rgb, u, v, train_test_splits_save_path)
