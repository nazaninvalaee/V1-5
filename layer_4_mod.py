import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Add, Multiply, BatchNormalization


def channel_attention(input_feature, ratio=8):
    """
    Implements a simple channel attention mechanism using global average
    and max pooling followed by shared dense layers.
    """
    channel = input_feature.shape[-1]

    shared_layer_one = layers.Dense(channel // ratio, activation='relu', use_bias=False)
    shared_layer_two = layers.Dense(channel, activation='sigmoid', use_bias=False)

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_two(shared_layer_one(avg_pool))

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_two(shared_layer_one(max_pool))

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Multiply()([input_feature, cbam_feature])

    return cbam_feature


def spatial_attention(input_feature):
    """
    Spatial Attention Module using average and max pooling along channels.
    """
    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_feature)

    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    attention = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid', use_bias=False)(concat)

    return Multiply()([input_feature, attention])


def residual_block(x, filters):
    """
    Residual block with optional projection shortcut and Batch Normalization.
    """
    shortcut = x

    # First Conv -> BN -> ReLU
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Second Conv -> BN
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Shortcut connection
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same', use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut) 

    x = Add()([shortcut, x])
    x = layers.Activation('relu')(x)

    return x


def multi_scale_conv(x, filters):
    """
    Multi-scale convolution block with 3x3, 5x5, and 7x7 filters, and Batch Normalization.
    """
    # 3x3 branch -> BN -> ReLU
    conv1 = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    
    # 5x5 branch -> BN -> ReLU
    conv2 = layers.Conv2D(filters, 5, padding='same', use_bias=False)(x)
    conv2 = BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    
    # 7x7 branch -> BN -> ReLU
    conv3 = layers.Conv2D(filters, 7, padding='same', use_bias=False)(x)
    conv3 = BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)

    concatenated = layers.Concatenate()([conv1, conv2, conv3])
    
    # 1x1 reduction -> BN -> ReLU
    reduced = layers.Conv2D(filters, 1, padding='same', use_bias=False)(concatenated)
    reduced = BatchNormalization()(reduced)
    reduced = layers.Activation('relu')(reduced)

    return reduced


def create_model(ensem=0, dropout_rate=0.2):
    """
    Creates an encoder-decoder model with attention and residual connections.
    """
    inp = layers.Input(shape=(256, 256, 1))

    # Encoder
    conv1 = residual_block(multi_scale_conv(inp, 16), 16)
    pool1 = layers.Dropout(dropout_rate)(layers.MaxPool2D(2)(conv1))

    conv2 = residual_block(multi_scale_conv(pool1, 32), 32)
    pool2 = layers.Dropout(dropout_rate)(layers.MaxPool2D(2)(conv2))

    conv3 = residual_block(multi_scale_conv(pool2, 64), 64)
    pool3 = layers.Dropout(dropout_rate)(layers.MaxPool2D(2)(conv3))

    conv4 = residual_block(multi_scale_conv(pool3, 128), 128)
    pool4 = layers.Dropout(dropout_rate)(layers.MaxPool2D(2)(conv4))

    # Bottleneck
    bottleneck = residual_block(multi_scale_conv(pool4, 256), 256)
    bottleneck = layers.Dropout(dropout_rate)(bottleneck)
    bottleneck = channel_attention(bottleneck)
    bottleneck = spatial_attention(bottleneck)

    # Decoder
    up4 = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu', use_bias=False)(bottleneck)
    up4 = residual_block(layers.Concatenate()([up4, conv4]), 128)

    up3 = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu', use_bias=False)(up4)
    up3 = residual_block(layers.Concatenate()([up3, conv3]), 64)

    up2 = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu', use_bias=False)(up3)
    up2 = residual_block(layers.Concatenate()([up2, conv2]), 32)

    up1 = layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu', use_bias=False)(up2)
    up1 = residual_block(layers.Concatenate()([up1, conv1]), 16)

    if ensem == 1:
        model = models.Model(inputs=inp, outputs=up1)
    else:
        final_output = layers.Conv2D(8, 1, padding='same', activation='sigmoid')(up1)
        model = models.Model(inputs=inp, outputs=final_output)

    return model
