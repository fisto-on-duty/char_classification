from keras.utils.np_utils import to_categorical
from extra_keras_datasets import emnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.ops.numpy_ops import np_config
from src.constants import IMAGE_SIZE

np_config.enable_numpy_behavior()


def load_dataset():
    (train_x, train_y), (test_x, test_y) = emnist.load_data(type='balanced')
    train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
    test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))

    train_x = tf.image.resize(train_x, IMAGE_SIZE)
    test_x = tf.image.resize(test_x, IMAGE_SIZE)
    # pad images to make them 32x32 instead of 28x28
    # train_x = tf.pad(train_x, [[0, 0], [2, 2], [2, 2], [0, 0]])
    # test_x = tf.pad(test_x, [[0, 0], [2, 2], [2, 2], [0, 0]])
    # one hot encode target values
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    # scale pixels
    train_norm = train_x.astype('float32')
    test_norm = test_x.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    train_norm_inverted = 1 - train_norm
    test_norm_inverted = 1 - test_norm

    return train_norm_inverted, train_y, test_norm_inverted, test_y


def rotate_by_90_deg_clockwise(image):
    image = tf.transpose(image, [1, 0, 2])
    return image


def preview_images(x_train, y_train, emnist_classes_mapping):
    # PREVIEW IMAGES
    plt.figure(figsize=(15, 4.5))
    for i in range(30):
        plt.subplot(3, 10, i+1)
        plt.imshow(x_train[i].reshape(IMAGE_SIZE), cmap=plt.cm.binary)
        print(f'{chr(emnist_classes_mapping[int(tf.argmax(y_train[i]))])}')
        plt.axis('off')
    plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
    plt.savefig('preview_images.png')


def preview_augmented_images(x_train, y_train, datagen, n_classes):
    # PREVIEW AUGMENTED IMAGES
    X_train3 = x_train[9, ].reshape((1, 32, 32, 1))
    Y_train3 = y_train[9, ].reshape((1, n_classes))
    plt.figure(figsize=(15, 4.5))
    for i in range(30):
        plt.subplot(3, 10, i+1)
        X_train2, Y_train2 = datagen.flow(X_train3, Y_train3).next()
        plt.imshow(X_train2[0].reshape(IMAGE_SIZE), cmap=plt.cm.binary)
        plt.axis('off')
        if i == 9:
            X_train3 = x_train[11, ].reshape((1, 32, 32, 1))
        if i == 19:
            X_train3 = x_train[18, ].reshape((1, 32, 32, 1))
    plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
    plt.savefig('preview_augmented_images.png')
