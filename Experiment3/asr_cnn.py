from __future__ import division, print_function, absolute_import
import tensorflow as tf


class CNNConfig(object):
    """CNN配置参数"""
    num_filters = 64  # 卷积核数目
    filter_sizes = [2,3,4,5]  # 卷积核尺寸
    hidden_dim = 32  # 全连接层神经元
    dropout_keep_prob = 1  # dropout保留比例
    learning_rate = 1e-3  # 学习率
    num_epochs = 1000  # 总迭代轮次
    batch_size = 128
    print_per_batch = 20
    save_tb_per_batch = 10


class ASRCNN(object):
    def __init__(self, config, width, height, num_classes):  # 20,100
        self.config = config
        self.input_x = tf.placeholder(tf.float32, [None, width, height], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # input_x = tf.reshape(self.input_x, [-1, height, width])
        input_x = tf.transpose(self.input_x, [0, 2, 1])
        pooled_outputs = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                print("conv-maxpool-%s" % filter_size)
                conv = tf.layers.conv1d(input_x, self.config.num_filters, filter_size, activation=tf.nn.relu)
                pooled = tf.reduce_max(conv, reduction_indices=[1])
                pooled_outputs.append(pooled)
        num_filters_total = self.config.num_filters * len(self.config.filter_sizes)  # 64*4
        pooled_reshape = tf.reshape(tf.concat(pooled_outputs, 1), [-1, num_filters_total])
        #pooled_flat = tf.nn.dropout(pooled_reshape, self.keep_prob)

        fc = tf.layers.dense(pooled_reshape, self.config.hidden_dim, activation=tf.nn.relu, name='fc1')
        fc = tf.contrib.layers.dropout(fc, self.keep_prob)
        #fc = tf.nn.relu(fc)
        # 分类器
        self.logits = tf.layers.dense(fc, num_classes, name='fc2')
        self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1, name="pred")  # 预测类别
        # 损失函数，交叉熵
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
        self.loss = tf.reduce_mean(cross_entropy)
        # 优化器
        self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        # 准确率
        correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))