from Experiment5.text_cnn import *
from Experiment5.utils import *
from keras.models import load_model
import os


def train():
    max_features = 5000
    maxlen = 400
    batch_size = 32
    embedding_dims = 50
    epochs = 10

    print('Build model......')
    model = TextCNN(maxlen, max_features, embedding_dims).get_model()
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    print('Train......')
    early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[early_stopping],
              validation_data=(x_val, y_val))

    if not os.path.exists("./saver"):
        os.makedirs("./saver")
    model.save("./saver/EmotionClassification_weights.h5")


def test():
    print('Test......')
    model = load_model("./saver/EmotionClassification_weights.h5")
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
    print('loss:', loss, 'accuracy:', accuracy)


if __name__ == '__main__':
    print('Loading data...')
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data_gen()
    print(len(x_train), 'train sequences')
    print(len(x_val), 'validation sequences')
    print(len(x_test), 'test sequences')

    train()
    test()