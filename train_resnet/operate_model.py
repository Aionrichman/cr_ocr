import os

import numpy as np
from keras.applications import *
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard


class DeepModel:

    def __init__(self, img_height, img_width, label_num, last_model=None):
        self.img_height = img_height
        self.img_width = img_width
        self.label_num = label_num

        if last_model:
            self.model = load_model(last_model)
        else:
            self.build_model()

    def build_model(self):
        base_model = ResNet50(include_top=False, weights=None, input_tensor=None,
                              input_shape=(self.img_height, self.img_width, 1), pooling=None, classes=None)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(4096, activation='relu')(x)
        y = Dense(self.label_num, activation='softmax')(x)

        self.model = Model(input=base_model.input, output=y)
        self.model.compile(optimizer=SGD(lr=0.001, momentum=0.9, decay=0.0001,nesterov=True),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

    def train_model(self, train_data, test_data, train_step, save_dir, log_dir):
        check_point = ModelCheckpoint(save_dir + os.sep + "deep_model-{epoch:02d}-{acc:.2f}.hdf5",
                                      monitor='val_acc',
                                      verbose=1,
                                      save_best_only=True,
                                      period=100)
        tensor_board = TensorBoard(log_dir=log_dir, write_grads=True)

        self.model.fit_generator(train_data, steps_per_epoch=128, epochs=train_step, validation_data=test_data,
                                 validation_steps=128, callbacks=[check_point, tensor_board])

    @staticmethod
    def test_model(model_path, img_path):
        dense_model = load_model(model_path)
        img = image.load_img(img_path, target_size=(32, 32), color_mode="grayscale")
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255

        predict_result_list = dense_model.predict(x)
        predict_result = np.where(predict_result_list == np.max(predict_result_list))

        return predict_result[1][0]
