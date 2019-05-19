import sys
import os
import matplotlib.pyplot as plt
from skimage import transform
from keras.preprocessing import image
from PIL import ImageEnhance
import numpy as np
import cv2

import logging

def array_to_img(array):
    from keras_preprocessing.image import array_to_img
    return array_to_img(array, 'channels_last')


def img_to_array(img):
    from keras_preprocessing.image import img_to_array
    return img_to_array(img, data_format='channels_last').astype(np.uint8)


def shift(x, wshift, hshift, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.1):
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = hshift * h
    ty = wshift * w
    return image.apply_affine_transform(x, tx=tx, ty=ty, row_axis=row_axis, col_axis=col_axis,
                                        channel_axis=channel_axis, fill_mode=fill_mode, cval=cval)


def rotate(img, degree):
    return transform.rotate(img, degree, resize=True, mode='edge', preserve_range=True).astype(np.uint8)


class ImageDataAugmentor(object):

    def __init__(self, scale_range, degree_range, mean, sigma, horizontal_range,
                 vertical_range, zoom_range, shear_intensity, brightness_range,
                 saturation_range, contrast_range, sharp_range):
        self.scale_range = scale_range
        self.degree_range = degree_range
        self.mean = mean
        self.sigma = sigma
        self.horizontal_range = horizontal_range
        self.vertical_range = vertical_range
        self.zoom_range = zoom_range
        self.shear_intensity = shear_intensity
        self.brightness_range = brightness_range
        self.saturation_range = saturation_range
        self.contrast_range = contrast_range
        self.sharp_range = sharp_range
        pass

    @staticmethod
    def random_rescale(img, scale_range):
        low = 1 - scale_range
        high = 1 + scale_range
        rnd_scale = np.random.uniform(low, high)

        return transform.rescale(img, rnd_scale, preserve_range=True).astype(np.uint8)

    @staticmethod
    def random_rotate(img, degree_range):
        rnd_degree = np.random.randint(-degree_range, degree_range)

        return rotate(img, rnd_degree)

    @staticmethod
    def random_gaussian_noise(img, mean=0.2, sigma=0.3):
        noise = np.random.normal(mean, sigma, img.shape)
        img = img + noise
        return img.astype(np.uint8)

    @staticmethod
    def random_shift(img, horizontal_range, vertical_range, fill_mode='nearest'):
        xshift = np.random.uniform(-horizontal_range, horizontal_range)
        yshift = np.random.uniform(-vertical_range, vertical_range)

        return shift(img, xshift, yshift, fill_mode=fill_mode)

    @staticmethod
    def random_crop(img, random_crop_size):
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - int(width * dx) + 1)
        y = np.random.randint(0, height - int(height * dy) + 1)

        return img[y:(y + int(height * dy)), x:(x + int(width * dx)), :]

    @staticmethod
    def random_zoom(img, low_limit, high_limit, fill_mode='nearest', cval=0.):
        return image.random_zoom(img, (low_limit, high_limit), row_axis=0, col_axis=1,
                                 channel_axis=2, fill_mode=fill_mode, cval=cval)

    @staticmethod
    def random_shear(img, intensity, fill_mode='nearest', cval=0.):
        return image.random_shear(img, intensity, row_axis=0, col_axis=1,
                                  channel_axis=2, fill_mode=fill_mode, cval=cval)

    @staticmethod
    def random_saturation(img, saturation_range):
        s_rate = np.random.uniform(saturation_range[0], saturation_range[1])
        pil_img = ImageEnhance.Color(array_to_img(img)).enhance(s_rate)
        return img_to_array(pil_img)

    @staticmethod
    def random_contrast(img, contrast_range):
        c_rate = np.random.uniform(contrast_range[0], contrast_range[1])
        pil_img = ImageEnhance.Contrast(array_to_img(img)).enhance(c_rate)
        return img_to_array(pil_img)

    @staticmethod
    def random_sharpness(img, sharp_range):
        rate = np.random.uniform(sharp_range[0], sharp_range[1])
        pil_img = ImageEnhance.Sharpness(array_to_img(img)).enhance(rate)
        return img_to_array(pil_img)

    @staticmethod
    def _random_activate():
        return np.random.uniform(0, 1) >= 0.5

    def random_augmentate(self, img):
        # if ImageDataAugmentor._random_activate():
        #     img = ImageDataAugmentor.random_rescale(img, self.scale_range)

        sel = np.random.randint(0, 3)
        # if sel == 0:
        #     img = ImageDataAugmentor.random_rotate(img, self.degree_range)

        if ImageDataAugmentor._random_activate():
            img = ImageDataAugmentor.random_crop(img, (self.horizontal_range, self.vertical_range))

        # if ImageDataAugmentor._random_activate():
        #     img = ImageDataAugmentor.random_zoom(img, self.zoom_range[0], self.zoom_range[1])

        # if ImageDataAugmentor._random_activate():
        #     img = ImageDataAugmentor.random_shear(img, self.shear_intensity)

        if ImageDataAugmentor._random_activate():
            img = ImageDataAugmentor.random_saturation(img,self.saturation_range)

        if ImageDataAugmentor._random_activate():
            img = ImageDataAugmentor.random_contrast(img, self.contrast_range)

        # if ImageDataAugmentor._random_activate():
        #     img = ImageDataAugmentor.random_sharpness(img, self.sharp_range)

        if ImageDataAugmentor._random_activate():
            img = image.random_brightness(img, self.brightness_range).astype(np.uint8)

        if self.mean is not None and self.sigma is not None \
                and ImageDataAugmentor._random_activate():
            img = ImageDataAugmentor.random_gaussian_noise(img, self.mean, self.sigma)

        return img


def augment_train_data(dataset_name):
    augmentor = ImageDataAugmentor(scale_range=0.3,
                                   degree_range=5,
                                   mean=None,
                                   sigma=None,
                                   horizontal_range=0.9,
                                   vertical_range=0.9,
                                   zoom_range=[0.8, 1.2],
                                   shear_intensity=5,
                                   brightness_range=[0.8, 1.2],
                                   saturation_range=[0.8, 1.2],
                                   contrast_range=[0.8, 1.2],
                                   sharp_range=[0.8, 1.2])

    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_data_dir = current_dir + os.sep + "dataset" + os.sep + dataset_name + os.sep + "train"
    label_dirs = os.listdir(train_data_dir)

    for label_dir in label_dirs:
        label_path = os.path.join(train_data_dir, label_dir)
        print(label_path)

        img_file_names = os.listdir(label_path)
        if len(img_file_names) > 300:
            continue

        for img_file_name in img_file_names:
            img_path = os.path.join(label_path, img_file_name)

            try:
                img = plt.imread(img_path)
            except OSError as exc:
                logging.warning(img_file_name + ":" + str(exc))
                os.remove(img_path)

            try:
                assert len(img.shape) == 3
            except AssertionError:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            for i in range(3):
                augment = img.copy()
                try:
                    augmented = augmentor.random_augmentate(augment)
                except ValueError as exc:
                    logging.warning(img_file_name + ":" + str(exc))
                    continue

                save_path = os.path.join(label_path, '%s_%d.png' % (os.path.splitext(img_file_name)[0], i))
                if np.mean(augmented) > 1:
                    plt.imsave(save_path, augmented)


def console_log():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
        datefmt='%Y-%m-%d %A %H:%M:%S',
        filename='log.log',
        filemode='a'
    )


if __name__ == '__main__':
    console_log()
    augment_train_data("cnftl_cps")


