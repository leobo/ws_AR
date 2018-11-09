import timeit

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.morphology as morphology
from matplotlib.patches import Circle
from scipy import ndimage
from skimage import filters, feature

from frameLoader import Frameloader


class Stipextractor(object):
    # def __init__(self, frames):
    #     self.frames = frames

    def __init__(self, frames, sigma, tau, scale, k):
        self.sigma = sigma
        self.tau = tau
        self.scale = scale
        self.k = k
        self.frames = np.array(frames)

    def draw_circle(self, image, X, Y, path, i):
        # Create a figure. Equal aspect so circles look circular
        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')

        # Show the image
        ax.imshow(image)

        # Now, loop through coord arrays, and create a circle at each x,y pair
        for xx, yy in zip(Y, X):
            circ = Circle((xx, yy), radius=1, color='red')
            ax.add_patch(circ)

        # Show the image
        # plt.show()

        plt.savefig(path + "/" + str(i) + ".png")
        plt.close()

    def detect_local_maxima(self, arr):
        # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
        """
        Takes an array and detects the troughs using the local maximum filter.
        Returns a boolean mask of the troughs (i.e. 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)
        """
        # arr = np.abs(arr)
        avg = np.average(arr)
        # arr[(arr > avg * 2)] = 0
        arr[(arr < avg)] = 0
        # define an connected neighborhood
        # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
        # neighborhood = morphology.generate_binary_structure(rank=len(arr.shape), connectivity=2)
        # apply the local minimum filter; all locations of minimum value
        # in their neighborhood are set to 1
        # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
        neighborhood = np.ones(shape=(3, 3, 3))
        local_max = (ndimage.maximum_filter(arr, footprint=neighborhood, mode='constant') == arr)
        # local_min is a mask that contains the peaks we are
        # looking for, but also the background.
        # In order to isolate the peaks we must remove the background from the mask.
        #
        # we create the mask of the background
        background = (arr == 0)
        #
        # a little technicality: we must erode the background in order to
        # successfully subtract it from local_min, otherwise a line will
        # appear along the background border (artifact of the local minimum filter)
        # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
        eroded_background = morphology.binary_erosion(
            background, structure=neighborhood, border_value=1)
        #
        # we obtain the final mask, containing only peaks,
        # by removing the background from the local_min mask
        detected_maxima = local_max ^ eroded_background
        return np.where(detected_maxima)

    def dollar_stip_extractor(self):
        """
        Extract the dollar's stip for self.frames.
        :return: the coordinates of the stips.
        """
        gaussian = []
        ev_list, od_list = self.get_gabor_filter(0.8, range(len(self.frames)))

        for i in range(len(self.frames)):
            img = filters.gaussian(self.frames[i], sigma=1.5)
            gaussian.append(img)
        gaussian = np.array(gaussian)

        t, x, y = gaussian.shape
        stips = np.zeros([t + t - 1, x, y])
        for r in range(x):
            for c in range(y):
                stips[:, r, c] = np.convolve(gaussian[:, r, c], ev_list, mode='full') + np.convolve(
                    gaussian[:, r, c], od_list, mode='full')
        coordinates = []
        print(int(t / 2))
        for i in range(int(t / 2), 2 * t - 1 - (int(t / 2)), 1):
            spatial_co = feature.peak_local_max(stips[i], min_distance=5, num_peaks=10)
            for co in spatial_co:
                coordinates.append((co[0], co[1], i - int(t / 2)))
        return gaussian, coordinates

    def get_gabor_filter(self, w, t_list):
        ev_list = []
        od_list = []
        for t in t_list:
            gabor_ev, gabor_od = self.gabor(w, t)
            ev_list.append(gabor_ev)
            od_list.append(gabor_od)
        return ev_list, od_list

    def gabor(self, w, t):
        return -np.cos(2 * np.pi * t * w) * np.exp((-t ** 2) / ((4 / w) ** 2)), -np.sin(
            2 * np.pi * t * w) * np.exp((-t ** 2) / ((4 / w) ** 2))

    def laptev_stip_extractor(self, method='k', k=0.05, eps=1e-6):
        """
        Extract the Laptev stips for all frames in self.frames
        :return: The 2D list contains the coordinates of stip.
        """
        self.frames = ndimage.gaussian_filter(self.frames, sigma=[self.sigma, self.sigma, self.tau], mode='constant', cval=0)
        Axx, Ayy, Att, Axy, Axt, Ayt = self.structure_tensor_3d(self.frames,
                                                                sigma=[self.sigma * self.scale, self.sigma * self.scale,
                                                                       self.tau * self.scale])

        detA = Axx * (Ayy * Att - Ayt ** 2) - Axy * (Axy * Att - Axt * Ayt) + Axt * (Axy * Ayt - Ayy * Axt)
        traceA = Axx + Ayy + Att

        if method == 'k' and k != 0.05:
            response = detA - self.k * traceA ** 2
        elif method == 'k' and k == 0.05:
            response = detA - k * traceA ** 2
        else:
            response = 2 * detA / (traceA + eps)

        coordinates = []
        for r in response:
            coordinates.append(feature.peak_local_max(r, min_distance=10, threshold_rel=0.2, num_peaks=30))
        # res = self.detect_local_maxima(response)
        # coordinates = []
        # for i in range(len(res[0])):
        #     coordinates.append([res[1][i], res[2][i], res[0][i]])
        return np.array(coordinates)

    def structure_tensor_3d(self, frames, sigma=[1, 1, 1], mode='constant', cval=0):
        """
        Extend the function skimage.feature.structure_tensor from 2d to 3d.
        :param frames: Input video (3d array)
        :param sigma: A 1d list contains standard deviation for the gaussian kernel at each axis.
        :return: Axx, Ayy, Att, Axy, Axt, Ayt
        """

        Ax, Ay, At = self.compute_derivatives_3d(frames, mode=mode, cval=cval)

        # weight the structure tensors
        Axx = ndimage.gaussian_filter(Ax * Ax, sigma, mode=mode, cval=cval)
        Ayy = ndimage.gaussian_filter(Ay * Ay, sigma, mode=mode, cval=cval)
        Att = ndimage.gaussian_filter(At * At, sigma, mode=mode, cval=cval)
        Axy = ndimage.gaussian_filter(Ax * Ay, sigma, mode=mode, cval=cval)
        Axt = ndimage.gaussian_filter(Ax * At, sigma, mode=mode, cval=cval)
        Ayt = ndimage.gaussian_filter(Ay * At, sigma, mode=mode, cval=cval)

        return Axx, Ayy, Att, Axy, Axt, Ayt

    def compute_derivatives_3d(self, frames, mode='constant', cval=0):
        """
        Extend the function skimage.feature._compute_derivatives from 2d to 3d.
        :param mode:
        :param cval:
        :return:
        """
        Ax = self.calculate_x_derivative(frames, mode, cval)
        Ay = self.calculate_y_derivative(frames, mode, cval)
        At = self.calculate_t_derivative(frames, mode, cval)
        return Ax, Ay, At

    def calculate_x_derivative(self, frames, mode='constant', cval=0):
        """
        Calculate the first order derivative for the second moment matrix in x dimension
        :param lsr: the second moment matrix
        :return: the x dimension first order dirivative for second moment matrix
        """
        x_derivative = []
        for f in frames:
            x_derivative.append(ndimage.sobel(f, axis=0, mode=mode, cval=cval))
        return np.array(x_derivative)

    def calculate_y_derivative(self, frames, mode='constant', cval=0):
        """
        Calculate the first order derivative for the second moment matrix in y dimension
        :param lsr: the second moment matrix
        :return: the y dimension first order dirivative for second moment matrix
        """
        y_derivative = []
        for f in frames:
            y_derivative.append(ndimage.sobel(f, axis=1, mode=mode, cval=cval))
        return np.array(y_derivative)

    def calculate_t_derivative(self, frames, mode='constant', cval=0):
        """
        Calculate the first order derivative for the second moment matrix in t dimension
        :param lsr: the second moment matrix
        :return: the t dimension first order dirivative for second moment matrix
        """
        t_derivative = np.zeros(shape=frames.shape)
        for y in range(frames.shape[2]):
            t_derivative[:, :, y] = ndimage.sobel(frames[:, :, y], axis=0, mode=mode, cval=cval)
        return np.array(t_derivative)

    # def create_second_moment_matrix(self, x, y, t):
    #     """
    #     Create the second moment matrix of linear scale representation with given x, y and t
    #     :param x: horizontal dimension
    #     :param y: vertical dimension
    #     :param t: temporal dimension
    #     :return: The moment matrix
    #     """
    #     v = [self.x_derivative[t, x, y], self.y_derivative[t, x, y], self.t_derivative[t, x, y]]
    #     m = np.outer(np.transpose(v), v)
    #     return np.multiply(self.get_gaussian_value_3d(self.sigma, self.tau, x, y, t), m)

    # def get_linear_scale_representation(self, sigma, tau, frames=None):
    #     """
    #     Build the linear scale representation by convolving all frames with the gaussian kernel
    #     :return: The scale representation
    #     """
    #     if frames == None:
    #         frames = self.frames
    #     return ndimage.gaussian_filter(np.float64(frames), sigma=(sigma, sigma, tau), mode='constant')

    # def create_gaussian_kernel(self, sigma, tau, t=None, x=None, y=None):
    #     """
    #     Create the 3d gaussian kernel with size x*y*t, the value for each component is given by the formula in
    #     et_gaussian_value
    #     :param sigma: variance
    #     :param tau: variance
    #     :return: the 3d gaussian kernel
    #     """
    #     if t == None and x == None and y == None:
    #         t, x, y = self.frames.shape
    #     gaussian_kernel = np.zeros(shape=(t, x, y))
    #     for k in range(t):
    #         for i in range(x):
    #             for j in range(y):
    #                 gaussian_kernel[k, i, j] = self.get_gaussian_value(sigma, tau, i, j, k)
    #     return gaussian_kernel

    # def create_gaussian_kernel_2d(self, sigma, x, y):
    #     gaussian_kernel = np.zeros(shape=(x, y))
    #     for i in range(x):
    #         for j in range(y):
    #             gaussian_kernel[i, j] = self.get_gaussian_value_2d(sigma, i, j)
    #     return gaussian_kernel

    # def create_gaussian_kernel_1d(self, sigma, x):
    #     gaussian_kernel = np.zeros(shape=(x))
    #     for i in range(x):
    #         gaussian_kernel[i] = self.get_gaussian_value_1d(sigma, i)
    #     return gaussian_kernel
    #
    # def get_gaussian_value_3d(self, sigma, tau, x, y, t):
    #     """Return the 3d Gaussian kernel.
    #     :return: the Gaussian kernel
    #     """
    #     return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2) - t ** 2 / (2 * tau ** 2)) / np.sqrt(
    #         2 * np.pi ** 3 * sigma ** 4 * tau ** 2)
    #
    # def get_gaussian_value_2d(self, sigma, x, y):
    #     return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    #
    # def get_gaussian_value_1d(self, sigma, x):
    #     return np.exp(-x ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)


if __name__ == '__main__':
    start = timeit.default_timer()
    # l = Frameloader(
    #     "/home/boy2/UCF101/UCF_101_dataset/UCF101_frames")
    s_path = '/Users/leo/Academic/PHD_videoSim/PHD_videoSim_dataset/UCF101/stip_show'
    l = Frameloader(
        '/Users/leo/Academic/PHD_videoSim/PHD_videoSim_dataset/UCF101/problemFrames/')
    f = l.load_frames()
    s = Stipextractor(f, 1, 30, 1, 0.03)

    # a, co = s.dollar_stip_extractor()
    co = s.laptev_stip_extractor()
    # for find local maxima
    # for i in range(len(f)):
    #     x = [c[0] for c in co if c[2] == i]
    #     y = [c[1] for c in co if c[2] == i]
    #     s.draw_circle(np.ndarray.tolist(s.frames[i]), x, y, s_path, i + 1)

    # for find 2d corner peak
    for i in range(len(co)):
        x = co[i][:,0]
        y = co[i][:,1]
        s.draw_circle(np.ndarray.tolist(s.frames[i]), x, y, s_path, i + 1)
    end = timeit.default_timer()
    print(end - start)

    # # testing
    # a = np.ones(shape=(5,5))
    # r = filters.gaussian(a, sigma=1)
    # r = filters.gabor(a, frequency=0.5)
    # # r = Stipextractor().get_linear_scale_representation(1, 1, a)
    # print(r)
    # print("---------------------")
    #
    # g = cv2.getGaborKernel(ksize=5, sigma=1, theta=0, lambd=5, gamma=0)
    # print(g)
