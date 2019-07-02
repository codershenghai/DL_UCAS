from Experiment3.utils import *
from Experiment3.asr_cnn import *


def train():
    # 将recordings文件夹下的音频文件按照7:2:1的比例分别分为训练集、验证集和测试集
    train_files, valid_files, test_files = load_files()

    print('读取训练文件......')
    train_features, train_labels = read_files(train_files)
    train_features = mean_normalize(train_features)

    print('读取验证文件......')
    valid_features, valid_labels = read_files(valid_files)
    valid_features = mean_normalize(valid_features)

    print('读取测试文件......')
    test_features, test_labels = read_files(test_files)
    test_features = mean_normalize(test_features)

    width = 20  # mfcc features
    height = 100  # (max) length of utterance
    classes = 10  # digits

    config = CNNConfig
    cnn = ASRCNN(config, width, height, classes)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    checkpoint_path = os.path.join('cnn_model', 'model.ckpt')
    tensorboard_train_dir = 'tensorboard/train'
    tensorboard_valid_dir = 'tensorboard/valid'

    if not os.path.exists(tensorboard_train_dir):
        os.makedirs(tensorboard_train_dir)
    if not os.path.exists(tensorboard_valid_dir):
        os.makedirs(tensorboard_valid_dir)
    tf.summary.scalar("loss", cnn.loss)
    tf.summary.scalar("accuracy", cnn.acc)
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(tensorboard_train_dir)
    valid_writer = tf.summary.FileWriter(tensorboard_valid_dir)

    total_batch = 0
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(train_features, train_labels)
        for x_batch, y_batch in batch_train:
            total_batch += 1
            feed_dict = feed_data(cnn, x_batch, y_batch, config.dropout_keep_prob)
            session.run(cnn.optim, feed_dict=feed_dict)
            if total_batch % config.print_per_batch == 0:
                train_loss, train_accuracy = session.run([cnn.loss, cnn.acc], feed_dict=feed_dict)
                valid_loss, valid_accuracy = session.run([cnn.loss, cnn.acc], feed_dict={cnn.input_x: valid_features,
                                                                                         cnn.input_y:  valid_labels,
                                                                                         cnn.keep_prob: config.dropout_keep_prob})
                print('Steps:' + str(total_batch))
                print('train_loss:' + str(train_loss) + ' train accuracy:' + str(train_accuracy) + '\tvalid_loss:' + str(
                        valid_loss) + ' valid accuracy:' + str(valid_accuracy))
            if total_batch % config.save_tb_per_batch == 0:
                train_s = session.run(merged_summary, feed_dict=feed_dict)
                train_writer.add_summary(train_s, total_batch)
                valid_s = session.run(merged_summary, feed_dict={cnn.input_x: valid_features,
                                                                 cnn.input_y: valid_labels,
                                                                 cnn.keep_prob: config.dropout_keep_prob})
                valid_writer.add_summary(valid_s, total_batch)

        saver.save(session, checkpoint_path, global_step=epoch)
    test_loss, test_accuracy = session.run([cnn.loss, cnn.acc],
                                           feed_dict={cnn.input_x: test_features,
                                                      cnn.input_y: test_labels,
                                                      cnn.keep_prob: config.dropout_keep_prob})
    print('test_loss:' + str(test_loss) + ' test accuracy:' + str(test_accuracy))


def test(path):
    features, label = read_test_wave(path)
    print('loading ASRCNN model...')
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('cnn_model/model.ckpt-999.meta')
        saver.restore(sess, tf.train.latest_checkpoint('cnn_model'))
        graph = tf.get_default_graph()
        input_x = graph.get_tensor_by_name("input_x:0")
        pred = graph.get_tensor_by_name("pred:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        for i in range(0, len(label)):
            feed_dict = {input_x: features[i].reshape(1, 20, 100), keep_prob: 1.0}
            test_output = sess.run(pred, feed_dict=feed_dict)

            print("=" * 15)
            print("真实lable: %d" % label[i])
            print("识别结果为:" + str(test_output[0]))
        print("\nCongratulation!")


if __name__ == '__main__':
    train()
    test("./numbersRec/test/")