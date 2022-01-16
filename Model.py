import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from Reprojector import Reprojector
from DataGenerator import DataGenerator

from Subnet import build_subnet_layer

img_size = 64

def build_hf_layer(input):
    sub_output = build_subnet_layer(input)

    conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(sub_output)
    conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv)
    conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(conv)

    return conv

def build_lf_layer(input):
    sub_output = build_subnet_layer(input)

    conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(sub_output)

    return conv

def build_conc_layer(input1, input2):
    contact = Concatenate(axis=3)([input1, input2])
    conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(contact)
    output = Conv2D(filters=1, kernel_size=(3, 3), padding='same')(conv)

    return output

def build_hf_model_only(input_shape):
    inputs = Input(shape=input_shape)

    sub_output = build_subnet_layer(inputs)

    conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(sub_output)
    conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv)
    output = Conv2D(filters=1, kernel_size=(3, 3), padding='same')(conv)

    model = Model(inputs, output, name='HF_only')

    return model

dg = DataGenerator()

def intensity_loss(h_pred, y_true):
    y_pred = dg.reprojector.intensity_reproject(img_array=dg.img, h_map_array=h_pred)
    return tf.math.sqrt(tf.reduce_sum(tf.multiply(1/(img_size**2),tf.square(y_true - y_pred))))

if __name__ == '__main__':
    input_shape = (256, 256, 1)
    model = build_hf_model_only(input_shape=input_shape)
    model.summary()

    output_shape = model.compute_output_shape(input_shape=(None, 256, 256, 1))
    print(output_shape)