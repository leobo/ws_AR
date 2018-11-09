import cv2
import matplotlib.pyplot as plt

path = "/home/boy2/UCF101/UCF_101_dataset/UCF101_frames/v_ApplyEyeMakeup_g01_c01/frame_1.jpg"
save_path = "/home/boy2/UCF101/2d_sruf_test"


class Surfextractor(object):
    def __init__(self, image, stips, hessian_threshold):
        self.interest_points = stips
        self.image = image
        self.hessian_threshold = hessian_threshold

    def key_points_create(self):
        """
        Create the opencv keypoints with the coordinates which are given in self.interest_points. All other arguments
        are set to:
        size = 1,
        angle = -1,
        response = 1,
        octave = 0,
        class_id = -1
        :return: opencv keypoints with given coordinates in points
        """
        return [cv2.KeyPoint(x, y, _size=1, _angle=-1, _response=0, _octave=0, _class_id=-1) for x, y in
                self.interest_points]

    def surf_extrate(self, key_points):
        """
        Calculate the surf feature for self.interest_points in the self.image with self.hessian_threshold
        :return: the surf features
        """

        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=self.hessian_threshold, extended=1, upright=True)
        # _, des = surf.compute(self.image, key_points)
        _, des = surf.detectAndCompute(self.image, None)
        while des is None:
            return []
        return des

    def key_surf_generate(self):
        """
        Create the opencv keypoint of self.interest_points calculate the surf features for them with according
        self.image and self.hessian_threshold
        :return: the surf features
        """
        kp = self.key_points_create()
        # if kp == []:
        #     print("No stips detected: [] is returned! ")
        #     return []
        des = self.surf_extrate(kp)
        return des


if __name__ == '__main__':
    image = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=1000, nOctaves=20, nOctaveLayers=1, upright=0, extended=1)
    kp, des = surf.detectAndCompute(image, None)
    img2 = cv2.drawKeypoints(image=image, keypoints=kp, outImage=image, color=[255,0,0])
    cv2.imwrite(save_path + '/frame_1.jpg', img2)


    print(des)
