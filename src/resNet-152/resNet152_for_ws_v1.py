import os
import time

import numpy as np
import tensorflow as tf

from frameLoader import Frameloader
from models.research.slim.nets import resnet_v1
import cv2

slim = tf.contrib.slim

framePath = ["/home/boy2/UCF101/ucf101_dataset/frames/jpegs_256", "/home/boy2/UCF101/ucf101_dataset/flows/tvl1_flow/u",
             "/home/boy2/UCF101/ucf101_dataset/flows/tvl1_flow/v"]
featureStorePath = ["/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_crop_v3",
                    "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop_v3/u",
                    "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_crop_v3/v"]
ws_image = ["/home/boy2/UCF101/ucf101_dataset/features/frame_ws_3"]
width = 224
height = 224
color_channels = 3


def flip(rgb, u, v):
    flip_frame_number = np.random.choice([0, 1], len(rgb))
    return np.array([f if b == 0 else np.fliplr(f) for f, b in zip(rgb, flip_frame_number)]), np.array(
        [f if b == 0 else -np.fliplr(f) + 255 for f, b in zip(u, flip_frame_number)]), np.array(
        [f if b == 0 else np.fliplr(f) for f, b in zip(v, flip_frame_number)])


def resize(frames, size):
    return np.array([cv2.resize(t, size, interpolation=cv2.INTER_CUBIC) for t in frames], dtype=np.float32)


def model():
    # load the resNet152 mode and the pre-trained weights and bias.
    input_layer = tf.placeholder(dtype=tf.float32, shape=[None, width, height, color_channels])

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        resNet152, end_points = resnet_v1.resnet_v1_152(input_layer,
                                                        num_classes=None,
                                                        is_training=False,
                                                        global_pool=True,
                                                        output_stride=None,
                                                        spatial_squeeze=True,
                                                        reuse=tf.AUTO_REUSE,
                                                        scope='resnet_v1_152')
    saver = tf.train.Saver()
    return input_layer, resNet152, end_points, saver

def cal_features(frames, sess, input_layer, resNet152):
    num_chunks = np.ceil(len(frames) / 128)
    chunks = np.array_split(np.array(frames), num_chunks)
    feature = []
    for c in chunks:
        # make input tensor for resNet152
        feature += list(np.reshape(sess.run(resNet152, feed_dict={input_layer: c}),
                                   newshape=[len(c), 2048]))
        # ep = sess.run(end_points, feed_dict={input_layer: c})
    return np.array(feature)


def gen_frame_feature_resNet152(rgb_fpath, rgb_spath, rgb_spath_flip, u_fpath, u_spath, u_spath_flip, v_fpath, v_spath,
                   v_spath_flip):
    """
    Read all video frames from framePath, then generate feature for each of them by resNet152. Store the features
    in featureStorePath.
    :return:
    """
    rgb_fl = Frameloader(rgb_fpath)
    u_fl = Frameloader(u_fpath)
    v_fl = Frameloader(v_fpath)

    # construct the model
    input_layer, resNet152, end_points, saver = model()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "/home/boy2/UCF101/src/resNet-152/resnet_v1_152.ckpt")
        # read all video frames and generate frame features by resNet152
        while len(rgb_fl.frame_parent_paths) != 0:
            # save start time
            start = time.time()

            # prepare the input frames
            video_name = rgb_fl.get_current_video_name()
            print("Current working on: ", video_name)
            rgb_frames = rgb_fl.load_frames()
            u_frames = u_fl.load_frames()
            v_frames = v_fl.load_frames()
            if len(rgb_frames) > len(u_frames) and len(rgb_frames) - 1 == len(u_frames):
                rgb_frames = rgb_frames[:-1]
            rgb_frames_flip, u_frames_flip, v_frames_flip = flip(rgb_frames, u_frames, v_frames)

            # resize the frames to (224, 224)
            rgb_frames = resize(rgb_frames, (width, height))
            u_frames = resize(u_frames, (width, height))
            v_frames = resize(v_frames, (width, height))
            rgb_frames_flip = resize(rgb_frames_flip, (width, height))
            u_frames_flip = resize(u_frames_flip, (width, height))
            v_frames_flip = resize(v_frames_flip, (width, height))

            if rgb_frames != [] and u_frames != [] and v_frames != [] and rgb_frames_flip != [] \
                    and u_frames_flip != [] and v_frames_flip != []:

                rgb_feature = cal_features(rgb_frames, sess, input_layer, resNet152)
                u_feature = cal_features(u_frames, sess, input_layer, resNet152)
                v_feature = cal_features(v_frames, sess, input_layer, resNet152)
                rgb_feature_flip = cal_features(rgb_frames_flip, sess, input_layer, resNet152)
                u_feature_flip = cal_features(u_frames_flip, sess, input_layer, resNet152)
                v_feature_flip = cal_features(v_frames_flip, sess, input_layer, resNet152)

                np.save(os.path.join(rgb_spath, video_name), rgb_feature)
                np.save(os.path.join(rgb_spath_flip, video_name), rgb_feature_flip)
                np.save(os.path.join(u_spath, video_name), u_feature)
                np.save(os.path.join(u_spath_flip, video_name), u_feature_flip)
                np.save(os.path.join(v_spath, video_name), v_feature)
                np.save(os.path.join(v_spath_flip, video_name), v_feature_flip)

            print(time.time() - start)


if __name__ == '__main__':
    framePath = ["/home/boy2/UCF101/ucf101_dataset/frames/jpegs_256",
                 "/home/boy2/UCF101/ucf101_dataset/flows/tvl1_flow/u",
                 "/home/boy2/UCF101/ucf101_dataset/flows/tvl1_flow/v"]
    featureStorePath = ["/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_resize",
                        "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_resize/u",
                        "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_resize/v"]
    featureStorePathFlip = ["/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_resize_flip",
                            "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_resize_flip/u",
                            "/home/boy2/UCF101/ucf101_dataset/frame_features/resNet_flow_resize_flip/v"]

    gen_frame_feature_resNet152(framePath[0], featureStorePath[0], featureStorePathFlip[0], framePath[1],
                                featureStorePath[1], featureStorePathFlip[1], framePath[2], featureStorePath[2],
                                featureStorePathFlip[2])