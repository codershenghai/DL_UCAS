from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Dropout, GlobalAveragePooling2D


img_size = 224
n_classes = 2
n_epochs = 30


def def_model():
    base_model = InceptionV3(weights='imagenet',
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

    return model 