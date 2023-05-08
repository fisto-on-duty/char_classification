import argparse
import os

import tensorflow as tf

from src.constants import SAVED_MODELS_PATH

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-i", "--input", help="path to directory with input image samples", required=True)


def launch_inference():
    args = arg_parser.parse_args()
    # imagenames_list = [filepath for filepath in os.listdir(args.images_dir_path) if filepath.endswith('.jpg') or
    #                    filepath.endswith('.png') or filepath.endswith('.jpeg')]

    with open('emnist_balanced_mapping.txt') as f:
        emnist_classes_mapping = dict([map(int, line.split()) for line in f])

    model = tf.keras.models.load_model(
        os.path.join(SAVED_MODELS_PATH, f"MobileNetV2_dropout0.2_leakyrelu_1024withregularizers.h5"))

    test_ds = tf.keras.utils.image_dataset_from_directory(args.images_dir_path, image_size=(32, 32),
                                                          batch_size=32, labels=None, color_mode='grayscale')
    # test_ds = tf.image.rgb_to_grayscale(test_ds)
    # for image_name in imagenames_list:
    #     path_to_image = os.path.join(args.images_dir_path, image_name)
    #
    #
    #     raw_img = tf.io.read_file(path_to_image)
    #     image = tf.io.decode_image(raw_img, dtype=tf.float32, channels=1)
    #
    #     # RESIZE ANY IMAGE IN 32x32
    #     # image = np.expand_dims(image, axis=-1)
    #     image = tf.image.resize(image, IMAGE_SIZE)

    predictions = model.predict(test_ds)
    predictions_classes = tf.math.argmax(predictions, axis=1).numpy().tolist()
    predictions_ascii = [chr(emnist_classes_mapping[key]) for key in predictions_classes]
    for path, pred_ascii in zip(test_ds.file_paths, predictions_ascii):
        print(f"{pred_ascii},{path}")

    # import matplotlib.pyplot as plt
    #
    # plt.figure(figsize=(10, 10))
    # for images in test_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         # plt.title(class_names[labels[i]])
    #         plt.axis("off")
    #     plt.savefig("squares.png")


if __name__ == '__main__':
    launch_inference()
