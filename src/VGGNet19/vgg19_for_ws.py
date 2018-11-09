import os

import cv2
import numpy as np
import tensorflow as tf

from VGGNet19.vgg19 import Vgg19
from frameLoader import Frameloader

import time

# def read_surfWS_tar(path):
#     surfWS_path = []
#     tar = []
#     surfWS_des = []
#     for (dirpath, dirnames, filenames) in os.walk(path):
#         surfWS_path += [os.path.join(dirpath, f) for f in filenames if f != '.DS_Store']
#         tar += [f.split()[0].split('_')[1] for f in filenames if f != '.DS_Store']
#     for surfWS in surfWS_path:
#         surfWS_des.append(np.load(surfWS))
#     return surfWS_des, tar

def main():
    frame_path = "/home/boy2/UCF101/UCF_101_dataset/UCF101_frames"
    fl = Frameloader(frame_path)
    fl.validate("/home/boy2/UCF101/VGGNet_frame_features")
    fl.frame_parent_paths = fl.shuffle(fl.frame_parent_paths)
    vgg = Vgg19()
    vgg.build()
    sess = tf.Session()
    for i in range(len(fl.frame_parent_paths)):
        start = time.time()
        video_name = fl.get_current_video_name()
        print("round ", i, "working on: ", video_name)
        temp = fl.load_frames()
        frames = [cv2.resize(t, (224, 224), interpolation=cv2.INTER_AREA) for t in temp]
        if frames != []:
            num_chunks = np.ceil(len(frames) / 100)
            chunks = np.array_split(np.array(frames), num_chunks)
            feature = []
            for c in chunks:
                feature += list(sess.run(vgg.fc8, feed_dict={vgg.rgb: c}))
            feature = np.array(feature)
            np.save(os.path.join("/home/boy2/UCF101/VGGNet_frame_features", video_name), feature)
            # time.sleep(5)
        print(time.time() - start)


if __name__ == '__main__':
    for i in range(10):
        main()
