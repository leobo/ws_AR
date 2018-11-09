from npyFileReader import Npyfilereader
from dvd.videoDescriptorDVD import Dvd_generator
from frameLoader import Frameloader
import numpy as np

low = -5
high = 5

def ind_vector_gen(size):
    """
    Generate a basis with dimensions size.
    :param size: The size of the basis
    :return: The generated basis
    """
    basis = np.zeros(size)
    ind = False
    while not ind:
        basis = np.random.randint(low=low, high=high, size=size)
        if np.linalg.det(basis) != 0 and -size[0] < np.linalg.det(basis) < size[0]:
            ind = True
        if sum(basis[0]) == 0 or sum(basis[1]) == 0:
            ind = False
    return basis

def calculate_DVD(frame_features_path, store_path, size):
    nr = Npyfilereader(frame_features_path)
    # nr.validate(store_path)
    basis = ind_vector_gen(size)
    for i in range(len(nr.npy_paths)):
        name, contents = nr.read_npys()
        dvd = Dvd_generator(name, contents, basis, store_path)
        dvd.cal_dvd()

if __name__ == '__main__':
    framePath = ["/home/boy2/UCF101/ucf101_dataset/features/resNet_crop"]
    featureStorePath = ["/home/boy2/UCF101/ucf101_dataset/features/"]
    for fp, fs in zip(framePath, featureStorePath):
        calculate_DVD(fp, fs, (2,2))
