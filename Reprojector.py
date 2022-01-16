import numpy as np
import tensorflow as tf

class Reprojector():

    def __init__(self, D, f, L):
        self.D = D
        self.L = L
        self.f = f


    def intensity_reproject(self, img_array, h_map_array):
        delta_phi = (2*np.pi*self.D*self.f*h_map_array)/(self.L - h_map_array)

        a = np.min(img_array)
        b = np.max(img_array) - a
        x_line = np.arange(img_array.shape[0]).reshape(1,img_array.shape[0])
        x = np.repeat(x_line,img_array.shape[1], axis=0)

        I = a + b*tf.math.cos(2*np.pi*self.f*x + delta_phi)

        return I