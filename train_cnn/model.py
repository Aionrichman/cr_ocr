import os


import tensorflow as tf


def weight_xavier(name, shape):
    initial = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    return initial


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class CNN:

    def __init__(self, sess, img_data, img_height=28, img_width=28, img_label=None, label_num=3982, training_flag=False):
        self.sess = sess
        self.img_data = img_data
        self.img_height = img_height
        self.img_width = img_width
        self.img_label = img_label
        self.label_num = label_num
        self.learning_rate = tf.placeholder(tf.float32, [])
        self.training_flag = training_flag

    def build_model(self, input_data):
        # 六层网络 + batch_norm
        w_conv1 = weight_xavier("w_conv1", [3, 3, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(input_data, w_conv1) + b_conv1)
        h_bn1 = tf.layers.batch_normalization(h_conv1, training=self.training_flag, name='bn1')

        w_conv2 = weight_xavier("w_conv2", [3, 3, 32, 32])
        b_conv2 = bias_variable([32])
        h_conv2 = tf.nn.relu(conv2d(h_bn1, w_conv2) + b_conv2)
        h_bn2 = tf.layers.batch_normalization(h_conv2, training=self.training_flag, name='bn2')

        w_conv3 = weight_xavier("w_conv3", [3, 3, 32, 64])
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(h_bn2, w_conv3) + b_conv3)
        h_bn3 = tf.layers.batch_normalization(h_conv3, training=self.training_flag, name='bn3')
        h_pool3 = max_pool_2x2(h_bn3)

        w_conv4 = weight_xavier("w_conv4", [3, 3, 64, 128])
        b_conv4 = bias_variable([128])
        h_conv4 = tf.nn.relu(conv2d(h_pool3, w_conv4) + b_conv4)
        h_bn4 = tf.layers.batch_normalization(h_conv4, training=self.training_flag, name='bn4')

        w_conv5 = weight_xavier("w_conv5", [3, 3, 128, 256])
        b_conv5 = bias_variable([256])
        h_conv5 = tf.nn.relu(conv2d(h_bn4, w_conv5) + b_conv5)
        h_bn5 = tf.layers.batch_normalization(h_conv5, training=self.training_flag, name='bn5')

        w_conv6 = weight_xavier("w_conv6", [3, 3, 256, 512])
        b_conv6 = bias_variable([512])
        h_conv6 = tf.nn.relu(conv2d(h_bn5, w_conv6) + b_conv6)
        h_bn6 = tf.layers.batch_normalization(h_conv6, training=self.training_flag, name='bn6')
        h_pool6 = max_pool_2x2(h_bn6)

        if self.label_num < 4000:
            w_fc1 = weight_xavier("w_fc1", [7 * 7 * 512, 4096])
            b_fc1 = bias_variable([4096])
            h_pool6_flat = tf.reshape(h_pool6, [-1, 7 * 7 * 512])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool6_flat, w_fc1) + b_fc1)
            h_bn7 = tf.layers.batch_normalization(h_fc1, training=self.training_flag, name='bn7')

            w_fc2 = weight_xavier("w_fc2", [4096, self.label_num])
            b_fc2 = bias_variable([self.label_num])
            label_conv = tf.matmul(h_bn7, w_fc2) + b_fc2
        else:
            w_fc1 = weight_xavier("w_fc1", [7 * 7 * 512, 8192])
            b_fc1 = bias_variable([8192])
            h_pool6_flat = tf.reshape(h_pool6, [-1, 7 * 7 * 512])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool6_flat, w_fc1) + b_fc1)
            h_bn7 = tf.layers.batch_normalization(h_fc1, training=self.training_flag, name='bn7')

            w_fc2 = weight_xavier("w_fc2", [8192, self.label_num])
            b_fc2 = bias_variable([self.label_num])
            label_conv = tf.matmul(h_bn7, w_fc2) + b_fc2


        self.saver = tf.train.Saver()
        self.predict = tf.argmax(tf.nn.softmax(label_conv), 1, output_type=tf.int32)

        if self.training_flag:
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.img_label,
                                                                                      logits=label_conv))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = tf.contrib.slim.learning.create_train_op(self.loss, optimizer, summarize_gradients=True)

            correct_prediction = tf.equal(self.predict, self.img_label)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            tf.summary.scalar('cross_entropy_loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

    def train(self, save_dir, train_step=3000):
        self.build_model(self.img_data)

        last_model = tf.train.latest_checkpoint(os.path.join(save_dir, "cnn_model"))
        if last_model:
            init_op = tf.local_variables_initializer()
            self.sess.run(init_op)
            self.saver.restore(self.sess, last_model)
        else:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            self.sess.run(init_op)

        summary_merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        lr_init = 0.001
        for i in range(train_step):
            lr_val = lr_init / (10 ** (i / 1000))
            _, summary_v, c_loss = self.sess.run([self.train_op, summary_merged, self.loss],
                                                feed_dict={self.learning_rate:lr_val})
            self.writer.add_summary(summary_v, global_step=i)

            if i % 10 == 0:
                train_accuracy = self.accuracy.eval()
                print("step {}, training accuracy {}, loss {}" .format(i, train_accuracy, c_loss))

                self.saver.save(self.sess,
                                save_dir + os.sep + "cnn_model" + os.sep + "cnn_model.ckpt",
                                global_step=i)

    def test(self):
        x = tf.placeholder("float", shape=[None, self.img_height, self.img_width, 1])
        self.build_model(x)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state("cnn_model")
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        res = self.sess.run(self.predict, feed_dict={x: [self.img_data]})

        return res

    def validate(self, batch_size, data_len):
        self.build_model(self.img_data)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        ckpt = tf.train.get_checkpoint_state("cnn_model")
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        all_accuracy = 0.0
        turn = int(data_len / batch_size)
        for i in range(turn):
            train_accuracy, c_loss = self.sess.run([self.accuracy, self.loss])
            all_accuracy += train_accuracy
            print("The accuracy of batch {} is {}, loss is {}".format(i, train_accuracy, c_loss))

        mean_accuracy = all_accuracy / turn
        print("The mean accuracy is {}".format(mean_accuracy))

