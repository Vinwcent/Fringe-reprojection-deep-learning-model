import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from scipy.interpolate import RectBivariateSpline
from Reprojector import Reprojector

img_size = 64

class DataGenerator():

    def __init__(self):
        self.gap = 5
        self.reprojector = Reprojector(5, 4, 5)
        self.set_gap(self.gap)
        return

    def gen_one(self):
        size = np.random.randint(4, 7)
        size = np.repeat(size, 2)

        loc = np.random.randint(0, 3)
        scale = 0.1 + abs(np.random.normal(0, scale=0.25))

        mat = np.abs(np.random.normal(loc=loc, scale=scale, size=size))
        mat = np.clip(mat, -200, 200)

        x = np.linspace(0, mat.shape[0], mat.shape[0])
        y = np.linspace(0, mat.shape[0], mat.shape[0])
        f = RectBivariateSpline(x=x, y=y, z=mat)

        x_gen = np.linspace(0, mat.shape[0], img_size)
        y_gen = np.linspace(0, mat.shape[0], img_size)
        r_gen = f(x_gen, y_gen)

        self.last_g = r_gen
        self.output_img = self.reprojector.intensity_reproject(img_array=self.img, h_map_array=self.last_g)
        self.output_img = tf.cast(tf.reshape(self.output_img, [img_size, img_size, 1]), dtype=tf.float32)

    def latest_surf(self):
        sns.set(style='white')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlim([0, 150])
        X, Y = np.meshgrid(range(0, self.last_g.shape[0]), range(0, self.last_g.shape[1]))

        ax.plot_surface(X, Y, self.last_g, cmap='jet')
        plt.show()

    def latest_img(self):
        i_array = self.reprojector.intensity_reproject(img_array=self.img, h_map_array=self.last_g)
        plt.imshow(i_array, cmap='gray')
        plt.show()


    def set_gap(self, gap):
        self.gap = gap

        b_line = np.zeros(shape=(img_size, gap))
        w_line = np.ones(shape=(img_size, gap))
        pattern = np.concatenate((b_line, w_line), axis=1)
        pattern = np.tile(pattern, (1, (img_size // 2) // gap + 1))
        img = pattern[:img_size, :img_size]
        self.img = img


    def compute_and_show(self):
        self.gen_one()

        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(self.img, cmap='gray')

        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        X, Y = np.meshgrid(range(0, self.last_g.shape[0]), range(0, self.last_g.shape[1]))
        ax2.plot_surface(X, Y, self.last_g, cmap='jet')

        ax3 = fig.add_subplot(1, 3, 3)
        i_array = self.reprojector.intensity_reproject(img_array=self.img, h_map_array=self.last_g)
        plt.imshow(i_array, cmap='gray')

        plt.show()