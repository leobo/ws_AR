import cv2
import numpy as np


class WsOverTD(object):

    def __init__(self, frames, weights):
        self.frames = frames
        self.weights = weights

    def sum_with_weights(self, frames=None, weights=None):
        """
        Calculate the weighted sum of every frames over temporal direction.
        Each frame is a gray image.
        The weights are given in weights parameter. The weights contains N t dimensional vectors, where t is the
        length of the video.
        :return: The 3d (N * W * H) array which contains the weighted sum of every frames with weights.
        """
        if frames is None and weights is None:
            frames = self.frames
            weights = self.weights

        t1, h, w = frames.shape
        t2, l = weights.shape

        if t1 != l:
            print("In function sum_with_weights: The video length are different with the weights length!")
            return None

        ws = np.zeros((t2, h, w))
        for l in range(t2):
            for x in range(h):
                ws[l, x, :] = np.dot(weights[l], frames[:, x, :])

        return np.swapaxes(np.swapaxes(ws, 0, 1), 1, 2)
        # return ws[0, :, :] + ws[1, :, :]


if __name__ == '__main__':
    f = cv2.imread("/home/boy2/UCF101/UCF_101_dataset/UCF101_frames/v_YoYo_g25_c05/frame_67.jpg")

    f = np.arange(27).reshape((3, 3, 3))
    print(f)
    w = np.array([[1, 1, 1], [2, 2, 2]])
    ws = WsOverTD(f, w)
    ws.sum_with_weights()
