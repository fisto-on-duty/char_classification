import datetime
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from src.constants import SAVED_MODELS_PATH, CHECKPOINTS_ROOT_PATH
from src.dataset_preparation import load_dataset
from src.models import setup_baseline_cnn, setup_efficientnet, setup_resnet, setup_mobilenet, \
    setup_baseline_cnn_even_more_dense_layers


def lr_schedule(x):
    """
    Returns a custom learning rate
    """
    learning_rate = 1e-3 * 0.95 ** x

    tf.summary.scalar('learning rate', data=learning_rate, step=1)
    return learning_rate


def launch_training():
    # with open('emnist_balanced_mapping.txt') as f:
    #     emnist_classes_mapping = dict([map(int, line.split()) for line in f])
    datagen = ImageDataGenerator(validation_split=0.1, rotation_range=10, zoom_range=0.1, width_shift_range=0.15,
                                 height_shift_range=0.15)
    val_datagen = ImageDataGenerator(validation_split=0.1)

    x_train, y_train, x_test, y_test = load_dataset()
    n_classes = y_train.shape[1]
    # preview_images(x_train, y_train, emnist_classes_mapping)
    # preview_augmented_images(x_train, y_train, datagen, n_classes)

    baseline_cnn_model = setup_baseline_cnn(n_classes)
    baseline_cnn_even_more_dense_layers = setup_baseline_cnn_even_more_dense_layers(n_classes)

    efficientnet_model = setup_efficientnet(n_classes)
    mobilenet_model = setup_mobilenet(n_classes)
    resnet_model = setup_resnet(n_classes)

    models = (baseline_cnn_even_more_dense_layers, )   # baseline_cnn_even_more_dense_layers

    epochs = 100
    batch_size = 32

    scalars_log_dir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(scalars_log_dir + "/metrics")
    file_writer.set_as_default()

    # annealer = LearningRateScheduler(lr_schedule)
    annealer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=1, min_lr=1e-05)

    # early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1)

    # x_train_90, x_val, y_train_90, y_val = train_test_split(x_train, y_train, test_size=0.1)
    for model in models:
        checkpoint_path = os.path.join(CHECKPOINTS_ROOT_PATH, f"{model.name}", "{epoch:03d}.ckpt")
        checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                                  # monitor='val_accuracy',
                                                                  monitor='val_loss',
                                                                  save_weights_only=True,
                                                                  save_best_only=True,
                                                                  verbose=1)
        # checkpoint_path = os.path.join(checkpoint_core_path, "{epoch:03d}")
        # checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True,
        #                                                           monitor='val_loss', verbose=1)
        fit_log_dir = f"logs/fit/{model.name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print(fit_log_dir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fit_log_dir, histogram_freq=1)

        # loss, test_accuracy = model.evaluate(x_test, y_test)
        # print(f"Untrained model, test_accuracy: {test_accuracy}, loss: {loss}")
        model.fit(datagen.flow(x_train, y_train, batch_size=batch_size, subset='training'),
                  epochs=epochs,
                  # steps_per_epoch=x_train_90.shape[0] // batch_size,
                  validation_data=val_datagen.flow(x_train, y_train, batch_size=batch_size, subset='validation'),
                  # validation_split=0.1,
                  callbacks=[checkpoints_callback, tensorboard_callback, annealer],
                  verbose=1)

        # reset the model, load the latest checkpoint, evaluate:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest_checkpoint)
        loss, test_accuracy = model.evaluate(x_test, y_test)
        print(f"{model.name}, test_accuracy: {test_accuracy}, loss: {loss}")

        model.save(os.path.join(SAVED_MODELS_PATH, f"{model.name}_best.h5"))


if __name__ == '__main__':
    launch_training()
