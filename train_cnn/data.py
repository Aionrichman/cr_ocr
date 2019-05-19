import os

import tensorflow as tf


class DataSet:

    def __init__(self, data_set_dir, batch_size=128, img_height=28, img_width=28, label_num=3982):
        self.data_set_dir = data_set_dir
        self.img_height = img_height
        self.img_width = img_width
        self.label_num = label_num

        self.build_input_data(batch_size)

    def read_data_set(self):
        img_path_list = []
        img_label_list = []

        for font_dir in os.listdir(self.data_set_dir):
            img_dir = os.path.join(self.data_set_dir, font_dir)
            for img_path in os.listdir(img_dir):
                img_path_list.append(os.path.join(img_dir, img_path))
                img_label_list.append(int(font_dir))

        return img_path_list, img_label_list

    def build_input_data(self, batch_size):
        img_path_list, img_label_list = self.read_data_set()

        img_paths = tf.convert_to_tensor(img_path_list, dtype=tf.string)
        img_labels = tf.convert_to_tensor(img_label_list, dtype=tf.int32)
        input_queue = tf.train.slice_input_producer([img_paths, img_labels])

        img_content = tf.read_file(input_queue[0])
        img_data = tf.image.decode_jpeg(img_content, channels=1)
        img_data = tf.image.resize_images(img_data, [self.img_height, self.img_width])
        img_data = img_data / 255.0

        img_label = input_queue[1]

        batch_img_data, batch_img_label = tf.train.batch([img_data, img_label], batch_size, 4, 512)

        self.data_num = len(img_label_list)
        self.batch_img_data = batch_img_data
        self.batch_img_label = batch_img_label


def run_pipe(func):
    def new_func(*args, **kwargs):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        res = func(*args, **kwargs)
        coord.request_stop()
        coord.join(threads)
        return res
    return new_func


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_set = DataSet(os.path.join(current_dir, "dataset/train"), label_num=11)
