import os

import numpy as np

from frameLoader import Frameloader
from No_use.stipReader import Stipreader
from No_use.surfExtractor import Surfextractor


class Framefeaturecalculator(object):
    def __init__(self, frame_path, stips_path, save_path):
        self.video_store_path = save_path
        self.frame_loader = Frameloader(frame_path)
        self.stip_reader = Stipreader(stips_path)
        self.current_video_name = self.set_current_video_name()

    def set_frame_num(self):
        """
        Set the frame number of the video (self.video_name)
        :return: the frame number
        """

    def set_current_video_name(self):
        """
        Set the video name of the frame feature calculator
        :return: the video name
        """
        return self.frame_loader.get_current_video_name()

    def cal_frame_feature(self):
        """
        :return:
        """
    def validate(self):
        # remove the path in frame loader and stip reader if surf already calculated
        self.frame_loader.validate(self.video_store_path)
        self.stip_reader.validate(self.video_store_path)

    def cal_save_features(self):
        """
        Calculate the frame features for all frames of the video (self.video_name).
        :return:
        """
        self.current_video_name = self.set_current_video_name()

        descriptors = []
        frames = self.frame_loader.load_frames()
        stips = self.stip_reader.load_single_stips()
        if len(frames) != len(stips):
            print("The number of frames is different with the number of stips")
            return None
        i = 0
        for f, s in zip(frames, stips):
            surf = Surfextractor(f, s, hessian_threshold=400)
            feature = surf.key_surf_generate()
            if feature != []:
                descriptors.append(feature)
            else:
                print(self.frame_loader.get_current_video_name() + ' frame: ' + str(i))
            i = i + 1
        self.save_feature(np.array(descriptors))
        return descriptors

    def save_feature(self, descriptors):
        """
        Save the descriptors under self.video_store_path/self.video_name with .npy format
        :param descriptors: the given all frames descriptors of one video
        """
        np.save(os.path.join(self.video_store_path, self.current_video_name), descriptors)


if __name__ == '__main__':
    f_path = "/Users/leo/Academic/PHD_videoSim/PHD_videoSim_dataset/UCF101/UCF101_frames"
    s_path = "/Users/leo/Academic/PHD_videoSim/PHD_videoSim_dataset/UCF101/stips_25fps_10each"
    save_path = "/Users/leo/Academic/PHD_videoSim/PHD_videoSim_dataset/UCF101/surf_25fps_10each"

    fl = Frameloader(f_path)
    s = Stipreader(s_path)
    frames = fl.load_frames(
        "/Users/leo/Academic/PHD_videoSim/PHD_videoSim_dataset/UCF101/UCF101_frames/v_BandMarching_g20_c03")
    stips = s.load_single_stips(
        "/Users/leo/Academic/PHD_videoSim/PHD_videoSim_dataset/UCF101/stips_25fps_10each/v_BandMarching_g20_c03.txt")
    print()
