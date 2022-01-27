# Fringe-reprojection-deep-learning-model

This program is part of a PSE project at ![](https://github.com/Vinwcent/Fringe-reprojection-deep-learning-model/blob/main/Presentations_pics/logo.png)

The model is made with tensorflow-macos 2.7, the presentation will be succinct since I'm still working to improve the model performance, here is a quick review so that you understand what it is and if it could be interesting for your project.

## Goal

The goal of this program is to find the bottom height profil of a drop held by its own vapor (Leidenfrost Effect)

![](https://github.com/Vinwcent/Fringe-reprojection-deep-learning-model/blob/main/Presentations_pics/Leiden.png)

## Method principle

To find this shape, we sent a fringe picture on the surface thanks to an optic system and get images like that.

![](https://github.com/Vinwcent/Fringe-reprojection-deep-learning-model/blob/main/Presentations_pics/fringe-example.png)

Thanks to the deep learning model in this folder, we can find the height profile.

## Model

It is simply a U-net neural network with some convolutional layer at the end to extract high frequency fringes informations that were missed without them.

![](https://github.com/Vinwcent/Fringe-reprojection-deep-learning-model/blob/main/Presentations_pics/unet.png)

## Training

We use a data generator that is based on the data generator of this [article](https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-20-32547&id=459819)

Datagenerator projects a given fringe model onto a randomly generated (with matrix interpolation) height-map.

![](https://github.com/Vinwcent/Fringe-reprojection-deep-learning-model/blob/main/Presentations_pics/projection.png)


© Vinwcent
