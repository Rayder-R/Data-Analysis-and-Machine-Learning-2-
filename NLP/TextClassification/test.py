import tensorflow as tf
# from tensorflow import keras
from keras import layers


layer = layers.BatchNormalization(input_shape=(100, 100, 3))
print(layer)
