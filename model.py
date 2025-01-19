import keras
import keras_cv
from config import CFG

def additional_encoder(x, filters):
    x = keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    return x

def build_segmentation_head():
    return keras.Sequential([
        keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.UpSampling2D(size=(4, 4), interpolation="bilinear"),
        keras.layers.Conv2D(
            filters=CFG.num_classes,
            kernel_size=1,
            use_bias=False,
            padding="same",
            activation="sigmoid",
            dtype="float32",
        ),
    ], name="segmentation_head")

def build_model():
    backbone = keras_cv.models.DeepLabV3Plus.from_preset(
        CFG.preset,
        input_shape=[*CFG.image_size, 3],
    )

    neck_layer_name = backbone.layers[-2].name
    out = backbone.get_layer(neck_layer_name).output

    additional_features = additional_encoder(backbone.input, 64)
    additional_features = keras.layers.MaxPooling2D()(additional_features)
    additional_features = additional_encoder(additional_features, 128)
    additional_features = keras.layers.MaxPooling2D()(additional_features)
    additional_features = additional_encoder(additional_features, 256)

    out = keras.layers.Concatenate()([out, additional_features])

    segmentation_head = build_segmentation_head()
    out = segmentation_head(out)

    model = keras.models.Model(inputs=backbone.input, outputs=out)
    return model