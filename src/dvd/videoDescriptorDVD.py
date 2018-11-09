import numpy as np
import os


class Dvd_generator(object):
    def __init__(self, name, features, basis, store_path):
        self.frame_features = features
        self.basis = basis
        self.name = name
        self.store_path = store_path

    def conv(self, features, vec):
        while features.shape[0] != 1:
            features = np.apply_along_axis(np.convolve, axis=0, arr=features, v=vec, mode='valid')
            if features.shape[0] != 1:
                features = np.subtract(features, np.mean(features, axis=0))
        return features

    def cal_dvd(self, features=None, basis=None):
        if features == None:
            features = self.frame_features
        if basis == None:
            basis = self.basis
        dvd = np.zeros(shape=(basis.shape[0], features.shape[1]))
        for i in range(dvd.shape[0]):
            dvd[i] = self.conv(features, basis[i])
        self.save_des(dvd)
        return dvd

    def save_des(self, descriptors):
        """
        Save the descriptors under self.video_store_path/self.video_name with .npy format
        :param descriptors: the given all frames descriptors of one video
        """
        np.save(os.path.join(self.store_path, self.name), descriptors)

