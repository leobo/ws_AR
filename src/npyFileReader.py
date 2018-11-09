import os

import numpy as np


class Npyfilereader(object):
    def __init__(self, path):
        self.npy_paths = self.gen_npy_paths(path)
        self.current_npy_name = self.set_current_npy_name()

    def set_current_npy_name(self):
        """
        Set the current working on npy file name
        :return: the npy file name
        """
        if len(self.npy_paths) != 0:
            return self.npy_paths[0].split('/')[-1]

    def dequeue_npy_path(self):
        """
        Dequeue the first npy file path
        """
        self.npy_paths = self.npy_paths[1:]

    def gen_npy_paths(self, path):
        """
        Generate the full paths for every npy files under the given path
        :param path: The parent path for the npy files
        :return: All paths for every npy files
        """
        npy_paths = []
        for (dirpath, dirnames, filenames) in os.walk(path):
            npy_paths += [os.path.join(dirpath, f) for f in filenames if os.path.splitext(f)[1] == '.npy']
        # sort the path list with alphabetically order
        npy_paths.sort()
        return npy_paths

    def read_single_npy(self, path):
        """
        Read and return the contents from single npy file
        :param path: The path of the npy file
        :return: The contents
        """
        return np.load(path)

    def read_npys(self):
        """
        Read and return the contents of only one npy file with path in self.npy_paths. Update the current working on npy
        file name and the self.npy_paths
        :return: The contents
        """
        npy_contents = self.read_single_npy(self.npy_paths[0])
        npy_name = self.current_npy_name
        self.dequeue_npy_path()
        self.current_npy_name = self.set_current_npy_name()
        return npy_name, npy_contents

    def validate(self, path):
        """
        Remove the element in self.npy_paths if it exists under the give path
        :param path: A path of dir
        """
        temp_paths = self.npy_paths[:]
        for p in self.npy_paths:
            name = p.split('/')[-1]
            temp_path = os.path.join(path, name.split('_')[1] + '_' + name)
            if os.path.exists(temp_path):
                temp_paths.remove(p)
        self.npy_paths = temp_paths
        self.set_current_npy_name()


if __name__ == '__main__':
    npy_path = "/Users/leo/Academic/PHD_videoSim/PHD_videoSim_dataset/UCF101/surf_25fps_10each"
    Npyfilereader(npy_path)
