# encoding: utf-8
import tensorflow as tf
from tensorflow import keras
import numpy as np
from UNet import UNet
import DataSet


# data process
def process(image, mask):
    image = tf.cast(image, dtype=tf.float32) / 255.
    mask = tf.cast(mask, dtype=tf.float32) / 255.
    return image, mask


# load dataset
image_train, image_var, image_test = DataSet.load_image()
mask_train, mask_var, mask_test = DataSet.load_mask()

db_train = tf.data.Dataset.from_tensor_slices((image_train, mask_train))
db_train = db_train.shuffle(1234).batch(120).map(process)

db_var = tf.data.Dataset.from_tensor_slices((image_var, mask_var))
db_var = db_var.shuffle(332).map(process)

db_test = tf.data.Dataset.from_tensor_slices((image_test, mask_test))
db_test = db_test.shuffle(145).map(process)


def main():
    learning_rate = 0.001
    Epoch = 400
    model = UNet()
    model.build(input_shape=(None, 572, 572, 3))
    optimizers = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)  # beta_1??

    for epoch in range(Epoch):

        for image, mask in enumerate(db_train):
            with tf.GradientTape() as tape:
                output = model(image)
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=output, logits=mask)
                loss_mean = tf.reduce_mean(loss)

                grads = tape.gradient(loss, model.trainable_variables)
                optimizers.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 20 == 0:
            print('loss_mean is ', loss)
