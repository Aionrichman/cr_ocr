import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras import backend as K


def freeze_session(sess, freeze_var_names=None, output_names=None, clear_devices=True):
    graph = sess.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(freeze_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_names,
            freeze_var_names
        )

        return frozen_graph


def export_graph(model_name, ex_model_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "models")
    model_path = os.path.join(model_dir, model_name)

    K.set_learning_phase(0)
    dense_model = load_model(model_path)

    sess = K.get_session()
    print(dense_model.input.op.name)
    print(dense_model.output.op.name)
    frozen_graph = freeze_session(sess, output_names=[dense_model.output.op.name])

    graph_path = os.path.join(model_dir, ex_model_name)
    with open(graph_path, 'wb') as f:
        f.write(frozen_graph.SerializeToString())


def import_graph(graph_name):
    with tf.gfile.GFile(graph_name, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="dense",
            op_dict=None,
            producer_op_list=None
        )

    return graph


def predict_by_graph(img_path, img_height=32, img_width=32):
    current_path = os.path.dirname(os.path.abspath(__file__))
    graph = import_graph(os.path.join(current_path, "cnftl_dense.pd"))
    x = graph.get_tensor_by_name('dense/input_1:0')
    y = graph.get_tensor_by_name('dense/dense_2/Softmax:0')

    with tf.Session(graph=graph) as sess:
        img = cv2.imread(img_path, 0)
        img_data = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_NEAREST)
        img_data = img_data.reshape([img_height, img_width, 1])
        img_data = img_data / 255.0

        res_index = sess.run(y, feed_dict={x: [img_data]})

    # res_chr = chr(int(ASCII_LIST[res_index[0]], 16))

    return res_index


if __name__ == '__main__':
    export_graph("deep_model-7800-0.98.hdf5", "cnftl_mobile.pd")

