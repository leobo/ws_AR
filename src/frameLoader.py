import os
import re

import cv2
import numpy as np


class Frameloader(object):
    def __init__(self, path):
        """
        Initial the frameLoader object
        :param path: the dir which contains frames of videos
        """
        self.frame_parent_paths = self.gen_frame_parent_paths(path)

    def validate(self, path):
        """
        Remove the element in self.frame_parent_paths if it exists under the give path
        :param path: A path of dir
        """
        temp_parent_paths = self.frame_parent_paths[:]
        for parent_path in self.frame_parent_paths:
            temp_path = os.path.join(path, parent_path.split('/')[-1] + '.npy')
            if os.path.exists(temp_path):
                temp_parent_paths.remove(parent_path)
        self.frame_parent_paths = temp_parent_paths

    def gen_frame_parent_paths(self, grand_parent_path):
        """
        Generate the path of the parent directories of the frames
        :grand_parent_path path: the grand parent path
        :return: the parent paths for the frames
        """
        frame_parent_paths = []
        for (dirpath, dirnames, filenames) in os.walk(grand_parent_path):
            frame_parent_paths += [os.path.join(dirpath, d) for d in dirnames]
        # sort the path list with alphabetically order
        frame_parent_paths.sort()
        return frame_parent_paths

    def shuffle(self, paths_list):
        """
        Randomly shuffle the elements in paths_list
        :param paths_list: path list
        :return: shuffled list
        """
        idx = np.random.randint(0, len(paths_list), len(paths_list))
        l = np.array(paths_list)[idx]
        return l

    def sort_numerically(self, file_list):
        """
        Sort the given list in the way that humans expect.
        :param file_list: the list contains file names
        :return: the numerically sorted file name list
        """
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        file_list.sort(key=alphanum_key)
        return file_list

    def load_single_frame(self, frame_path, mode):
        """
        Return the key frame with path frame_path, if the path is wrong, return None.
        :param frame_path: the path of the frame
        :return: the loaded frame
        """
        try:
            if mode == 'color':
                frame = cv2.imread(frame_path, 1)
            else:
                frame = cv2.imread(frame_path, 0)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            return None
        if frame is not None:
            return frame

    def load_frames(self, frame_parent_path=None, mode='color'):
        """
        Load all frames under the given parent directory
        :param frame_parent_path: the path of the parent directory
        :return: the frames under frame_parent_path
        """
        if frame_parent_path == None:
            frame_parent_path = self.frame_parent_paths[0]
            self.dequeue_frame_parent_path()
        frames = []
        frames_names = self.sort_numerically(os.listdir(frame_parent_path))
        for f in frames_names:
            if f != '.DS_Store':
                frame = self.load_single_frame(os.path.join(frame_parent_path, f), mode)
                if frame is not None:
                    frames.append(frame)
        return frames

    def dequeue_frame_parent_path(self):
        """
        Dequeue the first frame parent path in self.frame_parent_paths
        :return: None
        """
        self.frame_parent_paths = self.frame_parent_paths[1:]

    def get_current_video_name(self):
        """
        Get the current working on video
        :return:
        """
        return self.frame_parent_paths[0].split('/')[-1]


if __name__ == '__main__':
    fl = Frameloader("/Users/leo/Academic/PHD_videoSim/PHD_videoSim_dataset/UCF101/UCF101_frames")
    a = fl.load_frames()
    print()
