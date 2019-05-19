import os

import cv2
from keras.preprocessing.image import ImageDataGenerator


def rescale_img(dataset_path, img_height, img_width):
    font_dirs = os.listdir(dataset_path)
    for font_dir in font_dirs:
        abs_font_dir = os.path.join(dataset_path, font_dir)
        print(abs_font_dir)
        img_paths = os.listdir(abs_font_dir)
        for img_path in img_paths:
            abs_img_path = os.path.join(abs_font_dir, img_path)
            img = cv2.imread(abs_img_path, 0)
            img = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(abs_img_path, img)


def get_data_flow(data_dir, batch_size):
    data_gen = ImageDataGenerator(rescale=1. / 255,
                                  featurewise_center=True,
                                  featurewise_std_normalization=True,
                                  rotation_range=10,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  brightness_range=0.2)
    data_flow = data_gen.flow_from_directory(data_dir, target_size=(32, 32), color_mode='grayscale',
                                             class_mode='sparse', batch_size=batch_size, interpolation='bilinear')

    return data_flow
