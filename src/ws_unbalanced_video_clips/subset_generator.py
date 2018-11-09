import os
import re

import cv2
import numpy as np

u_dir = '/home/boy2/UCF101/ucf101_dataset/flows/tvl1_flow/u/'
v_dir = '/home/boy2/UCF101/ucf101_dataset/flows/tvl1_flow/v/'
rgb = '/home/boy2/UCF101/ucf101_dataset/frames/jpegs_256/'
out_put_dir = '/home/boy2/UCF101/ucf101_dataset/frame_features/selected_frames'


def select_frames():
    root, dirs, files = os.walk(rgb).__next__()
    dirs.sort()
    selected_frames = dict()
    for video in dirs:
        current_u_dir = u_dir + video + '/'
        current_v_dir = v_dir + video + '/'
        frame_list = [f for f in os.listdir(current_u_dir) if os.path.isfile(os.path.join(current_u_dir, f))]
        rgb_frames = []
        u_frames = []
        v_frames = []
        for frame in frame_list:
            current_u = current_u_dir + frame
            current_v = current_v_dir + frame
            frame_number = re.findall('\d+', frame)
            frame_number = int(frame_number[0])
            current_u_data = cv2.imread(current_u, cv2.IMREAD_GRAYSCALE)
            current_v_data = cv2.imread(current_v, cv2.IMREAD_GRAYSCALE)
            min_u = np.min(np.min(current_u_data, axis=0), axis=0)
            max_u = np.max(np.max(current_u_data, axis=0), axis=0)
            u_span = max_u - min_u
            min_v = np.min(np.min(current_v_data, axis=0), axis=0)
            max_v = np.max(np.max(current_v_data, axis=0), axis=0)
            v_span = max_v - min_v
            if v_span > 30 or u_span > 30:
                u_frames.append(frame_number)
                v_frames.append(frame_number)
                rgb_frames.append(frame_number)
                rgb_frames.append(frame_number + 1)
        u_frames.sort()
        v_frames.sort()
        rgb_frames.sort()
        rgb_frames = list(set(rgb_frames))
        np.save(os.path.join(out_put_dir, video), [rgb_frames, u_frames, v_frames])
    # return selected_frames


if __name__ == '__main__':
    select_frames()
