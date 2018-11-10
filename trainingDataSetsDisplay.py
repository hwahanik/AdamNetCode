from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import utils
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from keras.datasets import cifar100
import matplotlib.pyplot as plt

# Visualization of Keras datasets.

plt.interactive(True)

# Fashion-Mnist Classification
(xTrain, yTrain), (xTest, yTest) = fashion_mnist.load_data()
utils.show_shapes(xTrain, yTrain, xTest, yTest)
print('\n******************************\n')
utils.show_sample_image(xTrain, yTrain, cmap='gray')

# MNist dataset
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
utils.show_shapes(xTrain, yTrain, xTest, yTest)
print('\n******************************\n')
utils.show_sample_image(xTrain, yTrain)

# Sample cifar100 data set
(xTrain, yTrain), (xTest, yTest) = cifar100.load_data(label_mode='fine')
utils.show_shapes(xTrain, yTrain, xTest, yTest)
print('\n******************************\n')
utils.show_sample_image(xTrain, yTrain)