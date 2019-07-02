from Experiment1.model import *
from Experiment1.utils import *


def train_area():
    n_batch_size = 32
    n_epochs = 20
    n_classes = 26

    print("数据导入......")
    x_train, y_train = load_data("./dataset/train/area", n_classes)
    x_val, y_val = load_data("./dataset/val/area", n_classes)

    print("编译模型.......")
    model_area = def_model(n_classes=n_classes)
    sgd = Adam(lr=0.0003)
    model_area.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    print("训练模型.......")
    model_area.fit(x_train,
                   y_train,
                   batch_size=n_batch_size,
                   epochs=n_epochs,
                   verbose=1,
                   validation_data=(x_val, y_val))

    print("评估模型......")
    score, accuracy = model_area.evaluate(x_val, y_val, batch_size=n_batch_size)
    print('score:', score, 'accuracy:', accuracy)

    print("保存模型......")
    if not os.path.exists("./saver"):
        os.makedirs("./saver")
    model_area.save("./saver/LPR_area_weights.h5")


def train_letter():
    n_batch_size = 32
    n_epochs = 20
    n_classes = 34

    print("数据导入......")
    x_train, y_train = load_data("./dataset/train/letter", n_classes)
    x_val, y_val = load_data("./dataset/val/letter", n_classes)

    print("编译模型.......")
    model_area = def_model(n_classes=n_classes)
    sgd = Adam(lr=0.0003)
    model_area.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    print("训练模型.......")
    model_area.fit(x_train,
                   y_train,
                   batch_size=n_batch_size,
                   epochs=n_epochs,
                   verbose=1,
                   validation_data=(x_val, y_val))

    print("评估模型......")
    score, accuracy = model_area.evaluate(x_val, y_val, batch_size=n_batch_size)
    print('score:', score, 'accuracy:', accuracy)

    print("保存模型......")
    if not os.path.exists("./saver"):
        os.makedirs("./saver")
    model_area.save("./saver/LPR_letter_weights.h5")


def train_province():
    n_batch_size = 32
    n_epochs = 20
    n_classes = 31

    print("数据导入......")
    x_train, y_train = load_data("./dataset/train/province", n_classes)
    x_val, y_val = load_data("./dataset/val/province", n_classes)

    print("编译模型.......")
    model_area = def_model(n_classes=n_classes)
    sgd = Adam(lr=0.0003)
    model_area.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    print("训练模型.......")
    model_area.fit(x_train,
                   y_train,
                   batch_size=n_batch_size,
                   epochs=n_epochs,
                   verbose=1,
                   validation_data=(x_val, y_val))

    print("评估模型......")
    score, accuracy = model_area.evaluate(x_val, y_val, batch_size=n_batch_size)
    print('score:', score, 'accuracy:', accuracy)

    print("保存模型......")
    if not os.path.exists("./saver"):
        os.makedirs("./saver")
    model_area.save("./saver/LPR_province_weights.h5")


if __name__ == '__main__':
    train_area()
    train_letter()
    train_province()