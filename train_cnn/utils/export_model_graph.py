import os

import cv2
import tensorflow as tf
from model import CNN
from utils.config import *


def export_graph(path):
    img_height = 28
    img_width = 28
    # label_num = 6395
    label_num = 3982

    with tf.Session() as sess:
        x = tf.placeholder("float", shape=[None, img_height, img_width, 1])
        cnn = CNN(sess, x, label_num=label_num)
        cnn.build_model(x)

        ckpt = tf.train.get_checkpoint_state(path)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)

        graph = sess.graph
        graph_path = os.path.join(path, "cnftl12_cnn.pd")
        output_graph = tf.graph_util.convert_variables_to_constants(
            sess,
            graph.as_graph_def(),
            ['Softmax']
        )

        with open(graph_path, 'wb') as f:
            f.write(output_graph.SerializeToString())

    with graph.as_default():
        tf.import_graph_def(output_graph)
        # tf.summary.FileWriter(path, graph=graph)


def import_graph(graph_name):
    with tf.gfile.GFile(graph_name, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="cnn",
            op_dict=None,
            producer_op_list=None
        )

    return graph


def predict_by_graph(img_path, img_height=28, img_width=28):
    current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_path = os.path.join(current_path, "cnn_model")

    graph = import_graph(os.path.join(ckpt_path, "cnftl_cnn.pd"))
    x = graph.get_tensor_by_name('cnn/Placeholder:0')
    # y = tf.argmax(graph.get_tensor_by_name('cnn/Softmax:0'), 1, output_type=tf.int32)
    y = graph.get_tensor_by_name('cnn/Softmax:0')

    with tf.Session(graph=graph) as sess:
        img = cv2.imread(img_path, 0)
        img_data = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_LINEAR)
        img_data = img_data.reshape([img_height, img_width, 1])
        img_data = img_data / 255.0

        res_index = sess.run(y, feed_dict={x: [img_data]})

    # res_chr = chr(int(ASCII_LIST[res_index[0]], 16))

    return res_index


if __name__ == '__main__':
    current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    export_graph(os.path.join(current_path, "cnn_model"))


