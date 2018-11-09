from npyFileReader import Npyfilereader
import numpy as np
import os

if __name__ == '__main__':
    ucf_resNet_flow_path_1 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow_crop/u"
    ucf_resNet_flow_path_2 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow_crop/v"
    ucf_resNet_path = "/home/boy2/UCF101/ucf101_dataset/features/resNet_crop"

    ucf_resNet_flow_save_path_1 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow_crop_swap/u"
    ucf_resNet_flow_save_path_2 = "/home/boy2/UCF101/ucf101_dataset/features/resNet_flow_crop_swap/v"
    ucf_resNet_save_path = "/home/boy2/UCF101/ucf101_dataset/features/resNet_crop_swap"

    for p, s in zip([ucf_resNet_path, ucf_resNet_flow_path_1, ucf_resNet_flow_path_2],
                    [ucf_resNet_save_path, ucf_resNet_flow_save_path_1, ucf_resNet_flow_save_path_2]):
        fr = Npyfilereader(p)
        while len(fr.npy_paths) != 0:
            npy_name, npy_contents = fr.read_npys()
            npy_contents = np.swapaxes(npy_contents, 0, 1)
            np.save(os.path.join(s, npy_name), npy_contents)
            print(npy_name, "done")
