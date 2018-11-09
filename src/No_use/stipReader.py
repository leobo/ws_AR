import numpy as np
import os


class Stipreader(object):
    def __init__(self, path):
        self.stip_paths = self.gen_stips_paths(path)

    def validate(self, path):
        """
        Remove the element in self.stip_paths if it exists under the give path
        :param path: A path of dir
        """
        temp_stip_paths = self.stip_paths[:]
        for p in self.stip_paths:
            temp_path = os.path.join(path, p.split('/')[-1].split('.')[0] + '.npy')
            if os.path.exists(temp_path):
                temp_stip_paths.remove(p)
        self.stip_paths = temp_stip_paths

    def gen_stips_paths(self, path):
        """
        Generate the full path of all stips file under the given folder path.
        :param path: the given folder path which contains all stips files
        :return: the list of full path of stips
        """
        stips_paths = []
        for (dirpath, dirnames, filenames) in os.walk(path):
            stips_paths += [os.path.join(dirpath, d) for d in filenames if d != '.DS_Store']
        stips_paths.sort()
        return stips_paths

    def load_single_stips(self, stip_path=None):
        """
        Load all stips from a single txt file (for one video), then store them in a list of list.
        Each of the outer list contain all stips from one frame
        :return: the list of list which contains all stips of one video
        """
        if stip_path == None:
            stip_path = self.stip_paths[0]
            self.dequeue_stip_paths()
        return np.load(stip_path)

    def dequeue_stip_paths(self):
        """
        Dequeue the first element in self.stip_paths
        """
        self.stip_paths = self.stip_paths[1:]


if __name__ == '__main__':
    s = Stipreader("/Users/leo/Academic/PHD_videoSim/PHD_videoSim_dataset/UCF101/stips_25fps_10each")
    a = s.load_single_stips("/Users/leo/Academic/PHD_videoSim/PHD_videoSim_dataset/UCF101/stips_25fps_10each/v_CricketShot_g20_c01.txt")
    print()
