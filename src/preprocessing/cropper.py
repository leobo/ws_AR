from sklearn.feature_extraction.image import extract_patches_2d
from frameLoader import Frameloader
import numpy as np
import random
from skimage.io import imsave


class Cropper(object):
    def __init__(self, frames, size):
        self.frames = frames
        self.size = size

    def crop_image(self, image=None, size=None):
        """
        Randomly crop the given image with given size
        :param image: a 2d array,
        :param size: a tuple contain the W * H
        :return: a cropped image with size W*H
        """
        if image is None and size is None:
            image = self.frames
            size = self.size
        # return extract_patches_2d(image, size, max_patches=1)[0]
        w, h, _ = image.shape
        w_start = int((w-size[0])/2)
        h_start = int((h-size[1])/2)
        return image[w_start:w_start+size[0], h_start:h_start+size[1]]

    def flip(self, image):
        """
        Flip the given image
        :param image: a 2d array
        :return: the flipped image
        """
        p = random.choice([0, 1])
        if p == 1:
            return np.fliplr(image)
        else:
            return image

    def crop_flip(self):
        """
        Doing crop and flip(50%) for all frames
        :return: the processed frames
        """
        p_frames = []
        for p in self.frames:
            p_frames.append(self.crop_image(self.flip(p), self.size))
        return np.array(p_frames)

if __name__ == '__main__':
    loader = Frameloader("/home/boy2/UCF101/ucf101_dataset/frames/jpegs_256")
    images = loader.load_frames()
    cropper = Cropper(images, (224,224))
    image = images[0]
    res = cropper.crop_image(image, (224,224))
    fres = cropper.flip(res)
    imsave("/home/boy2/1.jpg", res)
    imsave("/home/boy2/2.jpg", fres)
    image = images[1]
    res = cropper.crop_image(image, (224, 224))
    fres = cropper.flip(res)
    imsave("/home/boy2/3.jpg", res)
    imsave("/home/boy2/4.jpg", fres)
    print()
