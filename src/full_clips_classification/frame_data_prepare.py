import numpy as np
import tensorflow as tf
import datetime, math, re, os
import cv2

from trainTestSamplesGen import TrainTestSampleGen
from sklearn import preprocessing, metrics
from full_clips_classification.combined_classifier import build_model

slim = tf.contrib.slim

ws_image = ["/home/boy2/UCF101/ucf101_dataset/features/frame_ws_3"]
IMG_WIDTH = 342
IMG_HEIGHT = 256
seg_num = 3

INPUT_WIDTH = 224
INPUT_HEIGHT = 224

RGB_CHANNELS = 3
FLOW_CHANNELS = 10
BATCH_SIZE = 1

dropout_rate = 0.9
num_unique_classes = 101
LEARNING_RATE = 0.001

total_epoch = 120
train_epoch = 1

clip_length = 5
num_cha = 9
min_len_video = 30
max_len_video = 1000
num_train_data = 9537
num_test_data = 3783
num_trans_matrices = 1
num_frame_features = 1
num_sample_train = 1
num_sample_test = 25


def flip(rgb, u, v):
    flip_frame_number = np.random.choice([0, 1], len(rgb))
    return np.array([f if b == 0 else np.fliplr(f) for f, b in zip(rgb, flip_frame_number)]), np.array(
        [f if b == 0 else -np.fliplr(f) + 255 for f, b in zip(u, flip_frame_number)]), np.array(
        [f if b == 0 else np.fliplr(f) for f, b in zip(v, flip_frame_number)])


def sort_numerically(file_list):
    """
    Sort the given list in the way that humans expect.
    :param file_list: the list contains file names
    :return: the numerically sorted file name list
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    file_list.sort(key=alphanum_key)
    return file_list


def gen_subvideo_v2(rgb, u, v, steps):
    if len(rgb) == len(u) + 1:
        rgb = rgb[:-1]
    data_splits_indeces = np.array_split(np.arange(len(rgb)), steps)
    rgb_indeces = [np.random.choice(d) for d in data_splits_indeces]
    rgb_splits = [rgb[i] for i in rgb_indeces]
    flow_indeces = []
    for i in rgb_indeces:
        if i - 5 < 0:
            start = 0
            stop = 10
        elif i + 5 > len(rgb) - 1:
            stop = len(rgb)
            start = stop - 10
        else:
            start = i - 5
            stop = i + 5
        flow_indeces.append((start, stop))

    u_splits = [u[index[0]:index[1]] for index in flow_indeces]
    v_splits = [v[index[0]:index[1]] for index in flow_indeces]

    return rgb_splits, u_splits, v_splits


def video_clips_selection(video_name, video_label, rgb_fpath, u_fpath, v_fpath):
    rgb_path = [os.path.join(rgb_fpath, n) for n in video_name]
    u_path = [os.path.join(u_fpath, n) for n in video_name]
    v_path = [os.path.join(v_fpath, n) for n in video_name]

    rgb, u, v, labels = list(), list(), list(), list()

    for rgb_p, u_p, v_p, l in zip(rgb_path, u_path, v_path, video_label):
        rgb_frame_p = sort_numerically([os.path.join(rgb_p, f) for f in os.listdir(rgb_p) if f.endswith('jpg')])
        u_frame_p = sort_numerically([os.path.join(u_p, f) for f in os.listdir(u_p) if f.endswith('jpg')])
        v_frame_p = sort_numerically([os.path.join(v_p, f) for f in os.listdir(v_p) if f.endswith('jpg')])

        rgb_frame_p, u_frame_p, v_frame_p = gen_subvideo_v2(rgb_frame_p, u_frame_p, v_frame_p, seg_num)
        rgb.append(rgb_frame_p)
        u.append(u_frame_p)
        v.append(v_frame_p)
    return rgb, u, v, video_label


def process_image(image, channel):
    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=channel)
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])
    # # Normalize
    # image = image * 1.0 / 127.5 - 1.0
    return image


def read_images(imagepaths, u_paths, v_paths, labels, batch_size):
    # Convert to Tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    u_paths = tf.convert_to_tensor(u_paths, dtype=tf.string)
    v_paths = tf.convert_to_tensor(v_paths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    # Build a TF Queue, shuffle data
    image, u, v, label = tf.train.slice_input_producer([imagepaths, u_paths, v_paths, labels], shuffle=True)

    rgb = tf.convert_to_tensor([process_image(image[i], RGB_CHANNELS) for i in range(seg_num)], dtype=tf.float32)
    flow = tf.convert_to_tensor([[[process_image(u[i, j], 1), process_image(v[i, j], 1)] for j in range(10)]
                                 for i in range(seg_num)], dtype=tf.float32)
    flow = tf.reshape(flow, [seg_num, 20, IMG_HEIGHT, IMG_WIDTH, 1])

    cropped_boxes = cropped_boxes_gen()
    # for rgb image
    cropped_image = list()
    for b in cropped_boxes:
        # original frame
        cropped_image.append(tf.image.crop_and_resize(rgb, boxes=[b for _ in range(seg_num)],
                                                      box_ind=[j for j in range(seg_num)], crop_size=[224, 224]))
        # flipped frame
        cropped_image.append(tf.image.crop_and_resize(tf.image.flip_left_right(rgb), boxes=[b for _ in range(seg_num)],
                                                      box_ind=[j for j in range(seg_num)],
                                                      crop_size=[224, 224]))
    rgb = tf.stack(cropped_image)

    # fro flow
    cropped_image = list()
    for b in cropped_boxes:
        # original frame
        original = tf.reshape(flow, [seg_num * 20, IMG_HEIGHT, IMG_WIDTH, 1])
        cropped_image.append(tf.image.crop_and_resize(original, boxes=[b for _ in range(seg_num * 20)],
                                                      box_ind=[j for j in range(seg_num * 20)],
                                                      crop_size=[224, 224]))
        # flipped frame
        flipped = tf.image.flip_left_right(original)
        cropped_image.append(
            tf.image.crop_and_resize(flipped, boxes=[b for _ in range(seg_num * 20)],
                                     box_ind=[j for j in range(seg_num * 20)],
                                     crop_size=[224, 224]))
    flow = tf.stack(cropped_image)

    label = tf.convert_to_tensor([label for _ in range(10)])

    # Create batches
    R, F, L = tf.train.batch([rgb, flow, label], batch_size=batch_size,
                             capacity=batch_size * 8,
                             enqueue_many=True,
                             allow_smaller_final_batch=True,
                             num_threads=4)
    return R, F, L


def test_accuracy(gt, predicts, num_test_samples, num_test_samples_per_video, num_trans_m):
    gt = np.concatenate(gt, axis=0)
    gt = np.reshape(gt,
                    [(num_test_samples_per_video * num_trans_m),
                     int(num_test_samples / (num_test_samples_per_video * num_trans_m))])
    gt = np.mean(gt, axis=0, dtype=np.int64, keepdims=False)
    predicts = np.concatenate(predicts, axis=0)
    predicts = np.reshape(predicts,
                          [(num_test_samples_per_video * num_trans_m),
                           int(num_test_samples / (num_test_samples_per_video * num_trans_m)),
                           num_unique_classes])
    pre_softmax = np.argmax(np.mean(predicts, axis=0, keepdims=False), axis=-1)
    return metrics.accuracy_score(gt, pre_softmax)


def cropped_boxes_gen():
    size = [256, 224, 192, 168]

    return [[0, 0, np.random.choice(size) / IMG_HEIGHT, np.random.choice(size) / IMG_WIDTH],
            [1 - np.random.choice(size) / IMG_HEIGHT, 0, 1, np.random.choice(size) / IMG_WIDTH],
            [0, 1 - np.random.choice(size) / IMG_WIDTH, np.random.choice(size) / IMG_HEIGHT, 1],
            [1 - np.random.choice(size) / IMG_HEIGHT, 1 - np.random.choice(size) / IMG_WIDTH, 1, 1],
            [np.random.choice(size) / IMG_HEIGHT / 2, np.random.choice(size) / IMG_WIDTH / 2,
             1 - np.random.choice(size) / IMG_HEIGHT / 2, 1 - np.random.choice(size) / IMG_WIDTH / 2]]


if __name__ == '__main__':
    dataset = 'ucf'
    framePath = ["/home/boy2/UCF101/ucf101_dataset/frames/jpegs_256",
                 "/home/boy2/UCF101/ucf101_dataset/flows/tvl1_flow/u",
                 "/home/boy2/UCF101/ucf101_dataset/flows/tvl1_flow/v"]
    frameStorePath = "/home/boy2/UCF101/ucf101_dataset/frame_features/cropped_frames"
    train_test_splits_save_path = "/home/boy2/UCF101/ucf101_dataset/features/testTrainSplits"

    encoder = preprocessing.LabelEncoder()

    if dataset == 'hmdb':
        tts = TrainTestSampleGen(ucf_path='', hmdb_path=train_test_splits_save_path)
    else:
        tts = TrainTestSampleGen(ucf_path=train_test_splits_save_path, hmdb_path='')

    # input data prepare
    t_rgb_clips, t_u_clips, t_v_clips, t_labels = video_clips_selection(tts.train_data_label[0]['data'][:10],
                                                                        encoder.fit_transform(
                                                                            tts.train_data_label[0]['label'])[:10],
                                                                        framePath[0], framePath[1], framePath[2])
    with tf.device('/gpu:0'):
        R, F, L = read_images(t_rgb_clips, t_u_clips, t_v_clips, t_labels, BATCH_SIZE)

        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
        mode = tf.placeholder(dtype=tf.bool, name='mode')

        # build the model
        logits = build_model(R, F, seg_num, False, mode, dropout_rate, num_unique_classes, BATCH_SIZE)

        # L = tf.reshape(tf.stack([L for _ in range(10)]), [BATCH_SIZE*10])
        # Calculate Loss (for both TRAIN and EVAL modes)
        onehot_labels = tf.one_hot(indices=tf.cast(L, tf.int32), depth=num_unique_classes)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

        accuracy_clips = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(input=logits, axis=1, output_type=tf.int32), L), tf.float32),
            name="acc")

        # Configure the Training Op (for TRAIN mode)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=1, total_num_replicas=None)
        train_op = optimizer.minimize(loss=loss)

        # tf.reset_default_graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            saver = tf.train.Saver()
            # saver = tf.train.import_meta_graph('/home/boy2/UCF101/ucf101_dataset/frame_features/checkpoint/model.ckpt.meta')
            # saver.restore(sess, tf.train.latest_checkpoint('/home/boy2/UCF101/ucf101_dataset/frame_features/checkpoint/'))

            # # test
            # shape = R.get_shape().as_list()
            # x = tf.reshape(R[0, 1], [1, IMG_HEIGHT, IMG_WIDTH, 3])
            # cropped_image = list()
            # for b in cropped_boxes:
            #     cropped_image.append(tf.image.crop_and_resize(x, boxes=b,
            #                                                   box_ind=[0],
            #                                                   crop_size=[224, 224]))
            # cropped_image.append(tf.image.resize_images(tf.image.central_crop(x, central_fraction=0.875), size=[224, 224]))
            # x = tf.stack(cropped_image)
            # x = tf.reshape(x, [6, 224, 224, 3])
            # res_image, ori = sess.run([x, R[0, 1]])
            # for i, j in zip(res_image, range(5)):
            #     cv2.imwrite('/home/boy2/Desktop/' + str(j) + '.jpg', i)
            # cv2.imwrite('/home/boy2/Desktop/o.jpg', ori)

            best_result = 0
            prev_loss = 10
            prev_acc = 0
            learning_rate = LEARNING_RATE

            exp_result = "/home/boy2/UCF101/ucf101_dataset/exp_results/res_for_1dconv_classifier_at_" + str(
                datetime.datetime.now()) + '.txt'
            print("Training start____________")
            for i in range(0, total_epoch + 1, train_epoch):
                # training
                mode = tf.estimator.ModeKeys.TRAIN
                total_train_loss = 0
                train_loss = 0
                # 269894 the # of training samples
                for j in range(1, int(math.ceil(num_train_data / BATCH_SIZE)) * train_epoch + 1):
                    _, loss_temp = sess.run([train_op, loss],
                                            feed_dict={'learning_rate:0': learning_rate, 'mode:0': mode})
                    total_train_loss += loss_temp
                    train_loss += loss_temp
                    if j % 100 == 0:
                        print("Setp", j, "The loss is", train_loss / 100)
                        train_loss = 0
                print("Training epoch", i, "finished, the avg loss is",
                      total_train_loss / j)

                # evaluation
                eval_acc_clips = 0
                eval_loss_clips = 0
                pre_logits = []
                gt = []
                print("______EVALUATION________")
                # 105164 the # for testing samples
                for j in range(1, int(math.ceil(num_test_data / BATCH_SIZE)) + 1):
                    loss_temp, accuracy_clips_temp, logits_temp, labels_temp = sess.run(
                        [loss, accuracy_clips, logits, L],
                        feed_dict={'learning_rate:0': learning_rate, 'mode:0': mode})
                    eval_acc_clips += accuracy_clips_temp
                    eval_loss_clips += loss_temp
                    pre_logits.append(logits_temp)
                    gt.append(labels_temp)
                eval_acc_clips /= j
                eval_loss_clips /= j

                print("Accuracy clips for evaluation is:", eval_acc_clips, "\n",
                      "loss is", eval_loss_clips)
                eval_acc = eval_acc_clips
                evaluation_loss = eval_loss_clips

                with open(exp_result, "a") as text_file:
                    text_file.writelines(
                        "Evaluation accuracy after training epoch %s is: %s \n" % (i * train_epoch, eval_acc))
                    # text_file.writelines(
                    #     "Softmax confusion matrix after training epoch %s is: \n" % (i * train_epoch))
                    # np.savetxt(text_file, cm_softmax, fmt='%s')
                    # text_file.writelines(
                    #     "Vote confusion matrix after training epoch %s is: \n" % (i * train_epoch,))
                    # np.savetxt(text_file, cm_logits, fmt='%s')
                if eval_acc < prev_acc:
                    # best_result = eval_acc
                    # if evaluation_loss > prev_loss + 0.01:
                    #     first = 0
                    learning_rate *= 0.1
                    print('The learning rate is decreased to', learning_rate)
                # if learning_rate <= 0.0000001:
                #     break
                if eval_acc > best_result:
                    best_result = eval_acc
                prev_loss = evaluation_loss
                prev_acc = eval_acc
                print("_________EVALUATION DONE___________")
                saver.save(sess, "/home/boy2/UCF101/ucf101_dataset/frame_features/checkpoint/model")
            coord.request_stop()
            coord.join(threads)
