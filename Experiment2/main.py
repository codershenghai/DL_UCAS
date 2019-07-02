import os
from Experiment2.utils import *
from Experiment2.InceptionV3 import *


def train():
    model = def_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit_generator(train_flow,
                        steps_per_epoch=300,
                        epochs=n_epochs,
                        verbose=1,
                        workers=8,
                        validation_data=test_flow,
                        validation_steps=300)
    if not os.path.exists("./saver"):
        os.makedirs("./saver")
    model.save("./saver/DogsVsCats_weights.h5")


if __name__ == '__main__':
    train()