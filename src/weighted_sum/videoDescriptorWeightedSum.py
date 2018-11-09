import os

import cv2
import numpy as np
import scipy.stats as stats


class Weightedsum(object):
    def __init__(self, name, features, store_path):
        self.frame_features = features
        self.name = name
        self.store_path = store_path

    def matrix_multiply(self, m1, m2):
        """
        Calculate the multiplication of matrices m1 (n*m) and m2(m*k)
        :param matrix1: the first matrix
        :param matrix2: the second matrix
        :return: the multiplication of

         m1 and m2
        """
        return np.matmul(m1, m2)

    def transformation_matrix_gen_cha(self, r, c, num_cha):
        """
        Generate the transformation matrix. The transformation matrix has dimension r*c and r<=c. Also, the transformation
        matrix contain r linear independent column vectors.
        :param r: the row number
        :param c: the column number
        :return: the transformation matrix
        """
        tram = []
        for i in range(int(r / num_cha)):
            temp = np.ones(shape=c)
            temp = np.vstack((temp, np.random.rand(num_cha - 1, c)))
            temp.sort()
            # temp = np.flip(temp, axis=1)
            # np.random.shuffle(temp)
            tram.append(temp)
        return np.array(tram).reshape([r, c])

    def ortho_matrix_gen(self, r, c, seed):
        np.random.seed(seed)
        temp = np.random.rand(c, c)
        temp, _ = np.linalg.qr(temp, mode='complete')
        # temp.sort()
        return temp[:r]

    # def co_attension_gen(self, feature):

    def attension_weights_gen(self, trans_m):
        attension_weights = []
        for row in trans_m:
            row_exp = np.exp(row)
            sum = np.sum(row_exp)
            attension_weights.append(row_exp / sum)
        return np.array(attension_weights)

    def transformation_matrix_gen_mix(self, r, c, seed):
        temp = np.ones(shape=c)
        loc = seed[-1]
        s = seed[0]
        temp = np.vstack((temp, np.random.rand(len(s), c)))
        temp.sort()
        for l in loc:
            u_dis = np.random.normal(l, 1, c)
            u_dis.sort()
            u_dis = np.abs(u_dis)
            if l > 0:
                temp = np.vstack((temp, u_dis))
            else:
                temp = np.vstack((temp, np.max(u_dis) - u_dis + 0.1))

        # temp = self.attension_weights_gen(temp)
        return temp

    def transformation_matrix_gen(self, r, c, seed):
        """
        Generate the transformation matrix. The transformation matrix has dimension r*c and r<=c. Also, the transformation
        matrix contain r linear independent column vectors.
        :param r: the row number
        :param c: the column number
        :return: the transformation matrix
        """
        # np.random.seed(seed)
        temp = np.ones(shape=c)
        for i in range(1, r, 1):
            starts = np.random.rand()
            steps = np.random.rand()
            temp = np.vstack((temp, np.arange(starts, starts + (c - 1) * steps + steps, steps)[:c]))
        # temp = np.flip(temp, axis=1)
        # np.random.shuffle(temp)
        return temp

        # # temp = np.ones(shape=c)
        # # for i in range(1, c, 1):
        # #     temp = np.vstack((temp, np.arange(i, i + (c - 1) * i + 1, step=i)+1))
        # np.random.seed(seed)
        # temp = np.random.rand(c, c)
        # temp, _ = np.linalg.qr(temp, mode='reduced')
        # m = np.min(temp)
        # temp += -m + 1
        # # np.random.shuffle(temp)
        # # temp.sort()
        # return temp[:r]

        # np.random.seed(seed)
        # temp = np.random.rand(r, c)
        # temp.sort()
        # return temp

    def transformation_matrix_gen_norm(self, r, c, loc):
        temp = np.ones(c)
        for l in loc:
            u_dis = np.random.normal(l, 1, c)
            u_dis.sort()
            u_dis = stats.norm.pdf(u_dis)
            temp = np.vstack((temp, u_dis))
        temp = self.attension_weights_gen(temp)
        return temp[:]

    def ws_descriptor_gen(self, r, save=True, trans_matrix=None):
        """
        Generate the video descriptor on top of frame descriptor (global or local) f_des by mapping certain elements in
        every frame features along time axis onto the plane which is described by trans_matrix.
        :param f_des: The frame features for a videos.
        :param r: The rank of the transformation matrix.
        :return: The video descriptor.
        """
        if trans_matrix is None:
            trans_matrix = self.transformation_matrix_gen(r, self.frame_features.shape[0])
        # feature = preprocessing.normalize(self.frame_features, axis=0, norm='l2')
        feature = self.frame_features
        temp = np.transpose(self.matrix_multiply(trans_matrix[:, :len(feature)], feature))
        # temp = self.post_processing(temp)
        if save:
            self.save_des(temp)
        return temp

        # temp = []
        # for r, c in zip(trans_matrix[:, :len(feature)], np.transpose(feature)):
        #     temp.append(np.dot(r, c))
        # return temp

    def ws_on_raw_data(self, r):
        trans_matrix = self.transformation_matrix_gen(r, self.frame_features.shape[0])
        temp = np.swapaxes(self.frame_features, 0, 2)
        temp = np.tensordot(temp, trans_matrix, axes=((2), (1)))
        self.save_des(temp)
        return temp

    def mean_descriptor_gen(self):
        m = np.mean(self.frame_features, axis=0)
        self.save_des(m)

    def post_processing(self, temp):
        post_processed_feature = []
        for frame_feature in temp:
            for line in frame_feature:
                post_processed_feature += [line]
        return np.array(post_processed_feature)

    def pre_processing(self):
        pre_processed_feature = []
        i = 0
        for frame_feature in self.frame_features:
            if frame_feature is None:
                return -1
            for line in frame_feature:
                pre_processed_feature += [line]
            i = i + 1
        self.frame_features = np.array(pre_processed_feature)

    def ws_descriptor_post_process(self, v_des):
        """
        For test.....
        :param v_des:
        :return:
        """
        post_pro_ws_des = []
        for v in v_des:
            temp = []
            for feature in v:
                temp = np.concatenate((temp, feature), axis=0)
            post_pro_ws_des.append(temp)
        return post_pro_ws_des

    def save_des(self, descriptors):
        """
        Save the descriptors under self.video_store_path/self.video_name with .npy format
        :param descriptors: the given all frames descriptors of one video
        """
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)
        np.save(os.path.join(self.store_path, self.name), descriptors)

    def save_as_image(self, data):
        cv2.imwrite(os.path.join(self.store_path, self.name + '.jpg'), data)
