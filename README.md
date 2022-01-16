# Fringe-reprojection-deep-learning-model

This program is part of a PSE project at ESPCI PARIS PSL.

The model is made with tensorflow 2.6

## Goal

The goal of this program is to find the bottom height profil of a drop held by its own vapor (Leidenfrost Effect)

![](https://github.com/Vinwcent/Fringe-reprojection-deep-learning-model/blob/main/Presentations_pics/Leiden.png)

## Method principle

To find this shape, we sent a fringed pictured on the surface thanks to an optic system and get images like that.

![](https://github.com/Vinwcent/Fringe-reprojection-deep-learning-model/blob/main/Presentations_pics/fringe-example.png)

Thanks to the deep learning model in this folder, we can find the height profile.

## Model

It is simply a U-net neural network with some convolutional layer at the end to extract high frequency fringes informations that were missed without them.

![](https://github.com/Vinwcent/Fringe-reprojection-deep-learning-model/blob/main/Presentations_pics/unet.png)
