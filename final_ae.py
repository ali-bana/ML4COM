# imports
import numpy as np

from keras.layers import Layer, PReLU, Conv2D, Activation, Conv2DTranspose, GaussianNoise, Lambda, Flatten, Reshape, \
    BatchNormalization, Reshape
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras import backend as K
import math

save_directory = 'saves'

# Load dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
# Divide data into test and validdation
X_test, X_validation, Y_test, Y_validation = train_test_split(X_test, Y_test, test_size=0.33, random_state=42)

# Normalizing dataset
X_train_norm = X_train / 255
X_test_norm = X_test / 255
X_validation_norm = X_validation / 255

k = 8 * 8 * 16
sqrtk = np.sqrt(k / 2)
c = k // 64
snr = 0
p = 1
std = np.sqrt(p / math.pow(10, snr / 10))
n = 32 * 32 * 3
np.random.seed(1000)
width = 32
height = 32
batch_size = 64
nb_epochs = 15
code_length = 128
print(std, std ** 2, 'k/n: ', k / (2 * n))



K.clear_session()
tf.set_random_seed(0)
np.random.seed(0)

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
        #     self.output_dim = output_dim
        self.sigma = sigma
        super(ChannelNoise, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        #         self.kernel = self.add_weight(name='kernel',
        #                                       shape=(input_shape[1], self.output_dim),
        #                                       initializer='uniform',
        #                                       trainable=True)
        self.inshape = input_shape
        super(ChannelNoise, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        #     l2_x = K.sqrt(K.dot(K.flatten(x),K.flatten(x)))
        #     h = K.random_normal(shape = (1), mean = 0, stddev = self.Hc)

        return x + K.random_normal(self.inshape[1:], mean=0, stddev=self.sigma)

    def compute_output_shape(self, input_shape):
        return input_shape


# Define model
model = Sequential()

# Encoder
model.add(Conv2D(16, (5, 5), padding='same', strides=2, input_shape=X_train.shape[1:]))
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2]))
# model.add(BatchNormalization())
model.add(Conv2D(32, (5, 5), padding='same', strides=2))
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2]))
# model.add(BatchNormalization())
model.add(Conv2D(32, (5, 5), padding='same', strides=1))
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2]))
# model.add(BatchNormalization())
model.add(Conv2D(32, (5, 5), padding='same', strides=1))
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2]))
# model.add(BatchNormalization())
model.add(Conv2D(c, (5, 5), padding='same', strides=1))
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2]))
# model.add(BatchNormalization(name='last'))
model.add(Flatten(name='last'))
model.add(ChannelNormalizer(sqrtk, name='normal'))
model.add(ChannelNoise(std, name='noise'))
model.add(Reshape([8, 8, c]))

# Decoder
model.add(Conv2DTranspose(32, (5, 5), padding='same', strides=1))
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2]))
# model.add(BatchNormalization())
model.add(Conv2DTranspose(32, (5, 5), padding='same', strides=1))
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2]))
# model.add(BatchNormalization())
model.add(Conv2DTranspose(32, (5, 5), padding='same', strides=1))
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2]))
# model.add(BatchNormalization())
model.add(Conv2DTranspose(16, (5, 5), padding='same', strides=2))
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2]))
# model.add(BatchNormalization())
model.add(Conv2DTranspose(3, (5, 5), padding='same', strides=2))
model.add(Activation('sigmoid'))


opt = keras.optimizers.Adam(lr=0.001)


def PSNR(y_true, y_pred):
    return 10 * K.log(K.max(y_true) ** 2 / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)


def schedule(epoch, lr):
    if epoch < 640:
        lr = 0.001
    else:
        lr = 0.0001
    return lr


# from google.colab import files
lrate = keras.callbacks.LearningRateScheduler(schedule, verbose=1)
chckpnt = keras.callbacks.ModelCheckpoint(save_directory + '/ae_0db_1_6_weights.{epoch}-{val_PSNR:.2f}.h5',
                                          monitor='val_PSNR', verbose=0, save_best_only=False,
                                          save_weights_only=True, mode='auto', period=1)
csv = keras.callbacks.CSVLogger(save_directory + '/ae_0db_1_6.log', separator=',', append=True)
opt = keras.optimizers.Adam(lr=0.001)
model.compile(loss='mse', optimizer=opt, metrics=[PSNR])

model.fit(X_train_norm, X_train_norm,
          batch_size=64,
          epochs=6,
          validation_data=(X_validation_norm, X_validation_norm),
          shuffle=True,
          callbacks=[lrate, csv, chckpnt])

