import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Add, Multiply
import layer_4_mod, layer_4_no_mod


def channel_attention(input_feature, ratio=8):
    """
    Implements a simple channel attention mechanism using global average
    and max pooling followed by shared dense layers.

    Args:
        input_feature (tf.Tensor): Input feature map of shape (H, W, C).
        ratio (int): Reduction ratio for the dense layer bottleneck.

    Returns:
        tf.Tensor: Channel-attended feature map.
    """
    channel = input_feature.shape[-1]

    shared_layer_one = layers.Dense(channel // ratio, activation='relu', use_bias=False, name='channel_attention_dense1')
    shared_layer_two = layers.Dense(channel, activation='sigmoid', use_bias=False, name='channel_attention_dense2')

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_two(shared_layer_one(avg_pool))

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_two(shared_layer_one(max_pool))

    cbam_feature = Add()([avg_pool, max_pool])
    attention_mechanism_output = Multiply(name='ensemble_channel_attention_output')([input_feature, cbam_feature])

    return attention_mechanism_output


def create_model(dropout_rate=0.2, num_classes=8, return_attention_map=False):
    """
    Creates a dual-branch ensemble model combining two backbone architectures
     and applying channel attention before final classification.

    Args:
        dropout_rate (float): Dropout rate for each sub-model.
        num_classes (int): Number of output classes for segmentation.
        return_attention_map (bool): If True, the model will also output
                                     the ensemble's channel attention map.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    model1 = layer_4_mod.create_model(ensem=1, dropout_rate=dropout_rate)
    model2 = layer_4_no_mod.create_model(ensem=1, dropout_rate=dropout_rate)

    inp = layers.Input(shape=(256, 256, 1))

    out1 = model1(inp)
    out2 = model2(inp)

    conc1 = layers.Concatenate()([out1, out2])
    
    ensemble_attention_output = channel_attention(conc1)
    
    attended_features = ensemble_attention_output

    conv2_for_gradcam = layers.Conv2D(16, 3, activation='relu', padding='same', name='gradcam_target_conv')(attended_features)
    conv2_output = layers.Conv2D(16, 3, activation='relu', padding='same')(conv2_for_gradcam)

    logits_output = layers.Conv2D(num_classes, 1, padding='same', name='logits_output')(conv2_output) 
    final_output_softmax = layers.Activation('softmax', name='segmentation_output')(logits_output)

    outputs = [final_output_softmax, logits_output]

    if return_attention_map:
        outputs.append(ensemble_attention_output)

    model = models.Model(inputs=inp, outputs=outputs)
        
    return model
