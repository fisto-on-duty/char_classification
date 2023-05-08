from keras.applications import EfficientNetV2S, EfficientNetB4, EfficientNetV2B0, Xception, ResNet50, MobileNetV2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Input, LeakyReLU, MaxPooling2D
from keras.optimizers import Adam
from keras import regularizers


def setup_baseline_cnn(n_classes, conv_dropout=0.25, dense_dropout=0.4):
    model = Sequential(name="baseline_cnn_different_dropout_rates_more_dense_layers_exact_units_count_match_lr_scheduler")

    model.add(Conv2D(32, kernel_size=3, activation=LeakyReLU(), input_shape=(32, 32, 1)))  # 'relu'
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=3, activation=LeakyReLU()))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation=LeakyReLU()))  # 14x14
    model.add(BatchNormalization())
    model.add(Dropout(conv_dropout))

    model.add(Conv2D(64, kernel_size=3, activation=LeakyReLU()))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation=LeakyReLU()))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation=LeakyReLU()))  # 7x7
    model.add(BatchNormalization())
    model.add(Dropout(conv_dropout))

    model.add(Conv2D(128, kernel_size=4, activation=LeakyReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(conv_dropout))

    model.add(Flatten())   # 512
    model.add(Dense(512, activation=LeakyReLU(), kernel_regularizer=regularizers.L2()))  # l2=0.01
    model.add(BatchNormalization())
    model.add(Dropout(dense_dropout))
    # model.add(Dense(256, activation=LeakyReLU()))
    # model.add(BatchNormalization())
    # model.add(Dropout(dense_dropout))
    model.add(Dense(128, activation=LeakyReLU(), kernel_regularizer=regularizers.L2()))
    model.add(BatchNormalization())
    model.add(Dropout(dense_dropout))

    model.add(Dense(n_classes, activation='softmax'))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    # print(model.summary())
    return model


def setup_baseline_cnn_even_more_dense_layers(n_classes, conv_dropout=0.3, dense_dropout=0.35):
    model = Sequential(name="baseline_cnn_even_more_dense_layers")

    model.add(Conv2D(32, kernel_size=3, activation=LeakyReLU(), input_shape=(32, 32, 1)))   # 30x30
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=3, activation=LeakyReLU()))  # 28x28
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation=LeakyReLU()))  # 14x14
    model.add(BatchNormalization())
    model.add(Dropout(conv_dropout))

    model.add(Conv2D(64, kernel_size=3, activation=LeakyReLU()))   # 12x12
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation=LeakyReLU()))   # 10x10
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation=LeakyReLU()))  # 5x5
    model.add(BatchNormalization())
    model.add(Dropout(conv_dropout))

    model.add(Conv2D(128, kernel_size=3, activation=LeakyReLU()))  # 3x3
    model.add(BatchNormalization())
    model.add(Dropout(conv_dropout))
    model.add(Conv2D(128, kernel_size=3, activation=LeakyReLU()))  # 1x1
    model.add(BatchNormalization())
    model.add(Dropout(conv_dropout))

    model.add(Flatten())
    model.add(Dense(256, activation=LeakyReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(dense_dropout))
    model.add(Dense(128, activation=LeakyReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(dense_dropout))

    model.add(Dense(n_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.005), loss="categorical_crossentropy", metrics=["accuracy"])

    # print(model.summary())
    return model


def setup_efficientnet(n_classes):
    model = Sequential(name='EfficientNetV2B0')

    efficientnet_model = EfficientNetV2B0(include_top=False, input_shape=(32, 32, 1), weights=None, classes=n_classes)
    model.add(efficientnet_model)
    model.add(Flatten())
    model.add(Dense(1280, activation=LeakyReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(640, activation=LeakyReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(n_classes, activation="softmax"))
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    # print(model.summary())
    return model


def setup_mobilenet(n_classes):
    model = Sequential(name='MobileNetV2')
    mobilenet_model = MobileNetV2(include_top=False, input_shape=(32, 32, 1), weights=None, classes=n_classes)
    model.add(mobilenet_model)
    model.add(Flatten())
    # model.add(Dense(1280, activation=LeakyReLU(), kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001),
    #           bias_regularizer=regularizers.L2(l2=0.001)))
    #     kernel_regularizer = regularizers.L1L2(l1=1e-5, l2=1e-4),
    #     bias_regularizer = regularizers.L2(1e-4),
    #     activity_regularizer = regularizers.L2(1e-5)
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    model.add(Dense(160, activation=LeakyReLU())) #kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001), bias_regularizer=regularizers.L2(l2=0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(n_classes, activation="softmax")) #, kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001)))
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())
    return model


def setup_resnet(n_classes):
    model = Sequential(name='resnet50')
    efficientnet_model = ResNet50(include_top=False, input_shape=(32, 32, 1), weights=None)  # ,classes=n_classes
    # print(efficientnet_model.summary())
    model.add(efficientnet_model)
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation="softmax"))
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    # print(model.summary())
    return model

