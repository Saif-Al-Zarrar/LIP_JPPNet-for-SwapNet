from tqdm import tqdm
import argparse
from datetime import datetime

# for debugging
import code


import os
import sys
import time
import scipy.misc
from scipy import sparse
import cv2
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from LIP_model import *
import argparse

N_CLASSES = 20
INPUT_SIZE = (256, 256)
# DATA_DIRECTORY = './datasets/examples'
# DATA_LIST_PATH = './datasets/examples/list/val.txt'
DATA_DIRECTORY = './datasets/outfit-transfer'
DATA_LIST_PATH = './datasets/outfit-transfer/tina_list.txt'
RESTORE_FROM = './checkpoint/JPPNet-s2'
OUTPUT_DIR = './output/parsing/val'

def main():
    parser = argparse.ArgumentParser(description="Evaluate parsing")
    parser.add_argument("-d", "--data_directory", help="Directory containing images.", default=DATA_DIRECTORY)
    parser.add_argument("-l", "--data_list", help=".txt file containing list of images to evaluate.", default=DATA_LIST_PATH)
    parser.add_argument("-o", "--output_directory", help="Directory containing images.", default=OUTPUT_DIR)
    parser.add_argument("-a", "--all_steps", action="store_true", help="Run all images instead of number of steps")
    parser.add_argument("-s", "--steps", type=int, help="Number of steps to run, instead of the whole directory")
    parser.add_argument("-v", "--visualize_step", type=int, help="How often to visualize")

    args = parser.parse_args()



    """Create the model and start the evaluation process."""

    # Create queue coordinator.
    coord = tf.train.Coordinator()
    h, w = INPUT_SIZE
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(args.data_directory, args.data_list, None, False, False, coord)
        image = reader.image
        image_rev = tf.reverse(image, tf.stack([1]))
        image_list = reader.image_list

    image_batch_origin = tf.stack([image, image_rev])
    image_batch = tf.image.resize_images(image_batch_origin, [int(h), int(w)])
    image_batch075 = tf.image.resize_images(image_batch_origin, [int(h * 0.75), int(w * 0.75)])
    image_batch125 = tf.image.resize_images(image_batch_origin, [int(h * 1.25), int(w * 1.25)])

    # Create network.
    with tf.variable_scope('', reuse=False):
        net_100 = JPPNetModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_075 = JPPNetModel({'data': image_batch075}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_125 = JPPNetModel({'data': image_batch125}, is_training=False, n_classes=N_CLASSES)


    # parsing net
    parsing_fea1_100 = net_100.layers['res5d_branch2b_parsing']
    parsing_fea1_075 = net_075.layers['res5d_branch2b_parsing']
    parsing_fea1_125 = net_125.layers['res5d_branch2b_parsing']

    parsing_out1_100 = net_100.layers['fc1_human']
    parsing_out1_075 = net_075.layers['fc1_human']
    parsing_out1_125 = net_125.layers['fc1_human']

    # pose net
    resnet_fea_100 = net_100.layers['res4b22_relu']
    resnet_fea_075 = net_075.layers['res4b22_relu']
    resnet_fea_125 = net_125.layers['res4b22_relu']

    with tf.variable_scope('', reuse=False):
        pose_out1_100, pose_fea1_100 = pose_net(resnet_fea_100, 'fc1_pose')
        pose_out2_100, pose_fea2_100 = pose_refine(pose_out1_100, parsing_out1_100, pose_fea1_100, name='fc2_pose')
        parsing_out2_100, parsing_fea2_100 = parsing_refine(parsing_out1_100, pose_out1_100, parsing_fea1_100, name='fc2_parsing')
        parsing_out3_100, parsing_fea3_100 = parsing_refine(parsing_out2_100, pose_out2_100, parsing_fea2_100, name='fc3_parsing')

    with tf.variable_scope('', reuse=True):
        pose_out1_075, pose_fea1_075 = pose_net(resnet_fea_075, 'fc1_pose')
        pose_out2_075, pose_fea2_075 = pose_refine(pose_out1_075, parsing_out1_075, pose_fea1_075, name='fc2_pose')
        parsing_out2_075, parsing_fea2_075 = parsing_refine(parsing_out1_075, pose_out1_075, parsing_fea1_075, name='fc2_parsing')
        parsing_out3_075, parsing_fea3_075 = parsing_refine(parsing_out2_075, pose_out2_075, parsing_fea2_075, name='fc3_parsing')

    with tf.variable_scope('', reuse=True):
        pose_out1_125, pose_fea1_125 = pose_net(resnet_fea_125, 'fc1_pose')
        pose_out2_125, pose_fea2_125 = pose_refine(pose_out1_125, parsing_out1_125, pose_fea1_125, name='fc2_pose')
        parsing_out2_125, parsing_fea2_125 = parsing_refine(parsing_out1_125, pose_out1_125, parsing_fea1_125, name='fc2_parsing')
        parsing_out3_125, parsing_fea3_125 = parsing_refine(parsing_out2_125, pose_out2_125, parsing_fea2_125, name='fc3_parsing')


    parsing_out1 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out1_100, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out1_075, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out1_125, tf.shape(image_batch_origin)[1:3,])]), axis=0)
    parsing_out2 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out2_100, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out2_075, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out2_125, tf.shape(image_batch_origin)[1:3,])]), axis=0)
    parsing_out3 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out3_100, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out3_075, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out3_125, tf.shape(image_batch_origin)[1:3,])]), axis=0)

    raw_output = tf.reduce_mean(tf.stack([parsing_out1, parsing_out2, parsing_out3]), axis=0)
    head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
    tail_list = tf.unstack(tail_output, num=20, axis=2)
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx]
    tail_list_rev[14] = tail_list[15]
    tail_list_rev[15] = tail_list[14]
    tail_list_rev[16] = tail_list[17]
    tail_list_rev[17] = tail_list[16]
    tail_list_rev[18] = tail_list[19]
    tail_list_rev[19] = tail_list[18]
    tail_output_rev = tf.stack(tail_list_rev, axis=2)
    tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))


    raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    # expand_dims to the beginning for the "batch" dimension
    before_argmax = tf.expand_dims(raw_output_all, dim=0)
    before_glasses = tf.slice(before_argmax, [0,0,0,0], [-1, -1, -1, 4])
    after_glasses = tf.slice(before_argmax, [0,0,0,5], [-1, -1, -1, -1])
    # this is now a 19-channel tensor
    without_glasses = tf.concat((before_glasses, after_glasses), axis=3)

    # # take out the background channel
    # seg_18 = before_argmax[:, :, :, 1:]
    # # convert to probability maps
    # seg_18_pmap = tf.nn.softmax(seg_18, axis=3)
    # # keep only the top 3 pmaps, because assume don't need more boundaries  than the top 3
    # seg_18_pmap_thin =


    # AJ: take the argmax of the channel dimension, to determine which clothing
    # label has the highest probabilitye
    argmaxed = tf.argmax(without_glasses, dimension=3)
    # argmax removed dim3, so add it back. Creates a 4d tensor, to make it batch x height x width x color
    pred_all = tf.expand_dims(argmaxed, dim=3)

    # Which variables to load.
    restore_var = tf.global_variables()
    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if RESTORE_FROM is not None:
        if load(loader, sess, RESTORE_FROM):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)


    # Iterate over training steps.
    num_steps = args.steps if args.steps else len(image_list) # added by AJ
    t = tqdm(range(num_steps), unit="img")
    for step in t:
        # removes the extension type
        img_id = os.path.splitext(image_list[step])[0]
        img_subpath = get_path_after_texture(img_id)

        # make output directory
        os.makedirs(os.path.join(args.output_directory, os.path.dirname(img_subpath)), exist_ok=True)

        t.set_description(img_subpath)

        # compute the output
        out = sess.run(pred_all)
        # create sparse matrix
        out_sparse = sparse.csc_matrix(np.squeeze(out))

        # seg_pmap = sess.run(seg_18_prob_map)
        # seg_pmap[seg_pmap < 0.05] = 0

        # save the numpy-array probability map to a file, so we can use it later
        fname = os.path.join(args.output_directory, img_subpath)
        sparse.save_npz(fname, out_sparse)
        # np.save(fname, out)

        if args.visualize_step and step % args.visualize_step == 0:
            msk = decode_labels(out)
            parsing_im = Image.fromarray(msk[0])
            parsing_im.save(f'{args.output_directory}/{img_subpath}_vis.png')

    coord.request_stop()
    coord.join(threads)

def get_path_after_texture(img_id):
    sep = os.path.sep
    path_elements = img_id.split(sep)
    tex_ind = path_elements.index("texture")
    path = sep.join(path_elements[tex_ind + 1:])
    return path


if __name__ == '__main__':
    main()


