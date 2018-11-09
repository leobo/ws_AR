import numpy as np

from frameLoader import Frameloader
from npyFileReader import Npyfilereader
from weighted_sum.videoDescriptorWeightedSum import Weightedsum


def calculate_weightedsum_fixed_len(train_data, dim, clip_len, flip=False):
    trans_m = None
    ws_des = []
    for data in train_data:
        if flip == True:
            data = np.flip(data, axis=0)
        ws = Weightedsum(None, data, None)
        if dim == 0:
            ws.mean_descriptor_gen()
        else:
            if trans_m is None:
                trans_m = ws.transformation_matrix_gen(dim, clip_len)
            ws_des.append(ws.ws_descriptor_gen(dim, save=False, trans_matrix=trans_m))
    return np.array(ws_des)


def calculate_weightedsum(frame_features_path, store_path, dim, flip=False):
    nr = Npyfilereader(frame_features_path)
    nr.validate(store_path)
    video_num = len(nr.npy_paths)
    for i in range(video_num):
        name, contents = nr.read_npys()
        # contents = contents[0::5]
        if flip == True:
            contents = np.flip(contents, axis=0)
        ws = Weightedsum(name, contents, store_path)
        if dim == 0:
            ws.mean_descriptor_gen()
        else:
            ws.ws_descriptor_gen(dim)


def calculate_ws_on_rawdata(frame_path, store_path, dim):
    fl = Frameloader(frame_path)
    fl.validate(store_path)
    while len(fl.frame_parent_paths) != 0:
        name = fl.get_current_video_name()
        frames = np.array(fl.load_frames(mode='color'))
        ws = Weightedsum(name, frames, store_path)
        if dim == 0:
            ws.mean_descriptor_gen()
        else:
            ws.ws_on_raw_data(dim)


if __name__ == '__main__':
    framePath = ["/home/boy2/UCF101/ucf101_dataset/frames/jpegs_256",
                 "/home/boy2/UCF101/ucf101_dataset/flows/tvl1_flow/u",
                 "/home/boy2/UCF101/ucf101_dataset/flows/tvl1_flow/v"]
    featureStorePath = ["/home/boy2/UCF101/ucf101_dataset/features/resNet_ws_image"]
    for fp, fs in zip(framePath, featureStorePath):
        calculate_ws_on_rawdata(fp, fs, 3)
