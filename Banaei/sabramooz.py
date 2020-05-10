# imports
import numpy as np
from tensorboardcolab import *
import keras
from keras.datasets import cifar10
from keras.models import Sequential,Model
from keras import backend as K
from keras.layers import Layer, PReLU, Conv2D, Activation, Conv2DTranspose , GaussianNoise,BatchNormalization,Conv1D,Flatten,Input,Dense,Reshape
from sklearn.model_selection import train_test_split
import tensorflow as tf
import math


class ChannelNormalizer(Layer):

    def __init__(self, sqrtk, **kwargs):
        #     self.output_dim = output_dim
        self.sqrtk = sqrtk
        super(ChannelNormalizer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ChannelNormalizer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return self.sqrtk * K.l2_normalize(x, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape


class ChannelNoise(Layer):

  def __init__(self, sigma, **kwargs):
    self.sigma = sigma
    super(ChannelNoise, self).__init__(**kwargs)

  def build(self, input_shape):
    self.inshape = input_shape
    super(ChannelNoise, self).build(input_shape)

  def call(self, x):
    return x + tf.random.normal(self.inshape[1:], mean = 0, stddev = self.sigma)

# Load dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
# Divide data into test and validdation
X_test, X_validation, Y_test,Y_validation  = train_test_split(X_test, Y_test, test_size=0.33, random_state=42)
# Normalizing dataset
X_train_norm = X_train/255
X_test_norm = X_test/255
X_validation_norm = X_validation/255

k = 8*8*16
sqrtk = np.sqrt(k/2)
c = k//64
snr = 0
p = 1
std = np.sqrt(0.5 / math.pow(10, snr/10))

#encoder part
input = Input(shape=(32,32,3))
conv_1 = Conv2D(16,(5,5),padding = 'same', strides = 2,activation='relu')(input)
conv_2 = Conv2D(32,(5,5),padding = 'same', strides = 2,activation='relu')(conv_1)
conv_3 = Conv2D(32,(5,5),padding = 'same', strides = 1,activation='relu')(conv_2)
conv_4 = Conv2D(32,(5,5),padding = 'same', strides = 1,activation='relu')(conv_3)
encoded = Conv2D(c,(5,5),padding = 'same', strides = 1,activation='relu')(conv_4)
encoded_flatten = Flatten()(encoded)

normalized = ChannelNormalizer(sqrtk)(encoded_flatten)
noize = ChannelNoise(std)(normalized)
noise_reshape = Reshape([8,8,c])(noize)

conv_0T = Conv2DTranspose(32,(5,5), padding = 'same', strides = 2,activation='relu')(noise_reshape)
conv_1T = Conv2DTranspose(32,(5,5), padding = 'same', strides = 1,activation='relu')(conv_0T)

z_mean = Conv2DTranspose(16,(5,5),padding = 'same', strides = 1,activation='relu')(conv_1T)
z_log_var = Conv2DTranspose(16,(5,5),padding = 'same', strides = 1,activation='relu')(conv_1T)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(K.shape(z_mean)[0],K.shape(z_mean)[1],K.shape(z_mean)[2],K.shape(z_mean)[3]), mean=0.,
                              stddev=1.0)
    return z_mean + K.exp(z_log_var / 2) * epsilon


from keras.layers import Input, Dense, Lambda,Reshape, Flatten
z = Lambda(sampling, output_shape=(16,16,16))([z_mean, z_log_var])


conv_2T = Conv2DTranspose(16,(5,5),padding = 'same', strides = 1,activation='relu')(z)
conv_3T = Conv2DTranspose(16,(5,5),padding = 'same', strides = 1,activation='relu')(conv_2T)
conv_4T = Conv2DTranspose(16,(5,5),padding = 'same', strides = 2,activation='relu')(conv_3T)
x_out = Conv2DTranspose(3,(5,5),padding = 'same', strides = 1,activation='sigmoid')(conv_4T)

vae = Model(input, x_out)
vae.summary()


def VAE_loss(x_origin,x_out):
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_origin- x_out), axis=[1, 2, 3]))
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    loss_sum = tf.reduce_mean(kl_loss + reconstruction_loss)
    return loss_sum

def PSNR(y_true, y_pred):
    return 10 * K.log(K.max(y_true) ** 2 / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)


def schedule(epoch, lr):
    #TODO compelete the scheduler
    if epoch < 640:
      lr = 0.001
    else:
      lr = 0.0001
    return lr


lrate = keras.callbacks.LearningRateScheduler(schedule, verbose=1)
opt = keras.optimizers.Adam(lr=0.001)

vae.compile(optimizer=opt, loss=VAE_loss, metrics=[PSNR])

vae.fit(X_train_norm, X_train_norm, shuffle=True, epochs=3000, batch_size=64,
        validation_data=(X_validation_norm, X_validation_norm), callbacks=[lrate])