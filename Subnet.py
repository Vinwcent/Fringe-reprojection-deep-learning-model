import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

img_size = 64

def conv_mini_block(input, n_filters_1, n_filters_2):
    conv = Conv2D(filters=n_filters_1, kernel_size=(3, 3), padding='same')(input)
    conv = BatchNormalization(trainable=False)(conv)
    conv = Activation('relu')(conv)

    conv = Conv2D(filters=n_filters_2, kernel_size=(3, 3), padding='same')(conv)
    conv = BatchNormalization(trainable=False)(conv)
    conv = Activation('relu')(conv)

    return conv

def encoder_mini_block(input, n_filters):
    conv = conv_mini_block(input=input, n_filters_1=n_filters, n_filters_2=2*n_filters)
    conv_pooled = MaxPool2D(pool_size=(2, 2))(conv)

    return conv, conv_pooled

def decoder_mini_block(input, skip_features, n_filters):
    up = Conv2DTranspose(filters=n_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(input)

    contact = Concatenate(axis=3)([up, skip_features])

    result = conv_mini_block(contact, n_filters_1=n_filters, n_filters_2=n_filters)

    return up

def build_subnet_layer(input):
    s1 = conv_mini_block(input, n_filters_1=1, n_filters_2=img_size // 4)
    p1 = MaxPool2D(pool_size=(2, 2))(s1)
    s2, p2 = encoder_mini_block(p1, n_filters=img_size // 4)
    s3, p3 = encoder_mini_block(p2, n_filters=img_size // 2)
    s4, p4 = encoder_mini_block(p3, n_filters=img_size)
    s5 = conv_mini_block(p4, n_filters_1=img_size*2, n_filters_2=img_size*4)

    d1 = decoder_mini_block(s5, s4, n_filters=img_size*2)
    d2 = decoder_mini_block(d1, s3, n_filters=img_size)
    d3 = decoder_mini_block(d2, s2, n_filters=img_size // 2)
    d4 = decoder_mini_block(d3, s1, n_filters=img_size // 4)

    return d4

def build_subnet(input_shape):
    inputs = Input(shape=input_shape)

    output = build_subnet_layer(inputs)

    subnet = Model(inputs, output, name='Subnet')

    return subnet

if __name__ == "__main__":
    input_shape = (img_size, img_size, 1)
    subnet = build_subnet(input_shape=input_shape)
    subnet.summary()





