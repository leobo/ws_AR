import os

import cv2
import skvideo


class Frameextractor(object):
    def __init__(self, videos_path):
        self.ind_video_path = self.gen_ind_video_path(videos_path)

    def gen_ind_video_path(self, videos_path):
        """
        Generate the relative paths for all videos(now didn't check if file a video) under the dir videos_path
        :param videos_path: the dir path
        :return: relative paths for all videos
        """
        v_path = []
        for (dirpath, dirnames, filenames) in os.walk(videos_path):
            v_path += [os.path.join(dirpath, f) for f in filenames if f.split('.')[-1] == 'avi']
        return v_path

    def extract_frames(self, v_path):
        """
        Extract all frames of video v_path
        :param v_path: the video path
        :return: the extracted frames
        """
        vidcap = cv2.VideoCapture(v_path)
        succ = True
        v_frames = []
        while succ == True:
            succ, frame = vidcap.read()
            if succ == True:
                v_frames.append(frame)
        return v_frames

        # vidcap = cv2.VideoCapture(v_path)
        # if not vidcap.isOpened():
        #     print("The error occurred when open video: " + v_path)
        #     return None
        #
        # v_frames = []
        # while vidcap.isOpened():
        #     success, image = vidcap.read()
        #     if success:
        #         v_frames.append(image)
        #     else:
        #         break
        # return v_frames

    def save_frames(self, v_frames, saved_path):
        """
        Save all frames in v_frames under dir saved_path, the name of the frames are their index in v_frames,
        the extension is '.jpg'.
        :param v_frames: the list contain the video frames
        :param saved_path: the path of dir
        :return: None
        """
        if not os.path.isdir(saved_path):
            os.makedirs(saved_path)

        count = 1
        for f in v_frames:
            cv2.imwrite(os.path.join(saved_path, "frame_%d.jpg") % count, f)
            count += 1

    def extract_and_save(self, saved_parent_path):
        """
        Extract all frames of all videos that paths in self.ind_video_path, then save the frames under dir save_path,
        each video has one dir.
        :param save_path: the saved dir path
        :return: None
        """
        for v_path in self.ind_video_path:
            label = v_path.split('/')[-2]
            video_name = v_path.split('/')[-1].split('.')[0]
            video_saved_path = os.path.join(saved_parent_path, label, video_name)
            if os.path.exists(video_saved_path):
                continue
            frames = self.extract_frames(v_path)
            self.save_frames(frames, video_saved_path)


if __name__ == '__main__':
    ext = Frameextractor("/home/boy2/UCF101/UCF_101_dataset/UCF_101_video")
    ext.extract_and_save("/home/boy2/UCF101/UCF_101_dataset/UCF_101_frames")
    print()
