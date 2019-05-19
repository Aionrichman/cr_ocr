import os
import argparse

import cv2
import tensorflow as tf

from data import DataSet, run_pipe
from model import CNN
from utils.export_model_graph import predict_by_graph
from utils.config import *


parser = argparse.ArgumentParser(description='')
parser.add_argument('--operation', dest='operation', default='train')
parser.add_argument('--data_set', dest='data_set', default='cnftl')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128)
parser.add_argument('--train_step', dest='train_step', type=int, default=3000)
parser.add_argument('--test_file', dest='test_file', default='4.png')

args = parser.parse_args()


@run_pipe
def train_model(sess, img_data, img_label, label_num, current_dir, training_step, img_height=28, img_width=28):
    cnn = CNN(sess,
              img_data,
              img_height=img_height,
              img_width=img_width,
              img_label=img_label,
              label_num=label_num,
              training_flag=True)
    cnn.train(current_dir, training_step)


def test_model(sess, img_path, label_num, img_height=28, img_width=28):
    img = cv2.imread(img_path, 0)
    img_data = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_LINEAR)
    img_data = img_data.reshape([img_height, img_width, 1])
    img_data = img_data / 255.0

    cnn = CNN(sess,
              img_data,
              img_height=img_height,
              img_width=img_width,
              label_num=label_num)
    res_index = cnn.test()

    res_chr = chr(int(ASCII_LIST[res_index[0]], 16))

    return res_chr


@run_pipe
def validate_model(sess, img_data, img_label, label_num, batch_size, data_len, img_height=28, img_width=28):
    cnn = CNN(sess,
              img_data,
              img_label=img_label,
              img_height=img_height,
              img_width=img_width,
              label_num=label_num)
    cnn.validate(batch_size, data_len)


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    label_list = os.listdir(current_dir + os.sep + "dataset" + os.sep + args.data_set + os.sep + "train")
    label_num = len(label_list)
    print(label_num)

    with tf.Session() as sess:
        if args.operation == 'train':
            train_set = DataSet(current_dir + os.sep + "dataset" + os.sep + args.data_set + os.sep + "train",
                                batch_size=args.batch_size,
                                label_num=label_num)
            train_model(sess, train_set.batch_img_data, train_set.batch_img_label, label_num, current_dir, args.train_step)
        elif args.operation == 'test':
            # predict_chr = test_model(sess, os.path.join("./testset/", args.test_file), label_num)
            # print(predict_chr)

            print(predict_by_graph(os.path.join("./testset/", args.test_file)))
        elif args.operation == 'validate':
            test_set = DataSet(current_dir + os.sep + "dataset" + os.sep + args.data_set  + os.sep + "test",
                               batch_size=args.batch_size,
                               label_num=label_num)
            validate_model(sess, test_set.batch_img_data, test_set.batch_img_label, label_num, args.batch_size, test_set.data_num)
        else:
            print("error argument")
