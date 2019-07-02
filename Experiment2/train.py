import keras
from DogsVsCats import data_move
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D

img_size = 20
n_classes = 2
n_epochs = 30


def main():
    img_size = 224
    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       pooling=None,
                       input_shape=(img_size, img_size, 3),
                       classes=n_classes)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(n_classes, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit_generator(data_move.train_flow,
                        steps_per_epoch=300,
                        epochs=n_epochs,
                        verbose=1,
                        workers=32,
                        validation_data=data_move.test_flow,
                        validation_steps=500,
                        callbacks=[tbCallBack])
    model.save("./saver/DogsVsCats_weights.h5")


# def pred_data():
#     model_path = "./saver/model_area.yaml"
#     weights_path = "./saver/weight_area.h5"
#     with open(model_path) as yaml_file:
#         load_model_yaml = yaml_file.read()
#     model = model_from_yaml(load_model_yaml)
#     model.load_weights(weights_path)
#
#     sgd = Adam(lr=0.0003)
#     model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#     path = r"C:\Users\78247\PycharmProjects\LPR\dataset\val\province\30\debug_chineseMat829.jpg"
#     img = image.load_img(path, target_size=image_size)
#     img_array = image.img_to_array(img)
#     x = np.expand_dims(img_array, axis=0)
#     x = preprocess_input(x)
#     result = model.predict_classes(x, verbose=0)
#     print(result[0])

if __name__ == '__main__':
    main()