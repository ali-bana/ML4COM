import numpy as np
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras import backend as K
from keras.layers import Layer, PReLU, Conv2D, Activation, Conv2DTranspose , GaussianNoise,BatchNormalization,Conv1D
from sklearn.model_selection import train_test_split
import tensorflow as tf
import math
import numpy as np
from keras.models import Model
from keras.layers import Layer, PReLU, Conv2D, Activation, Conv2DTranspose , GaussianNoise,Lambda, Flatten, Dropout,BatchNormalization,Reshape
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras import backend as K
import math

from keras.layers import Input, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Flatten
#%%
#make a 'saves' directory beside code to save callbacks and logs
save_directory = 'saves1/'

#%%

# Load dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
# Divide data into test and validdation
X_test, X_validation, Y_test,Y_validation  = train_test_split(X_test, Y_test, test_size=0.33, random_state=42)

# Normalizing dataset
X_train_norm = X_train/255
X_test_norm = X_test/255
X_validation_norm = X_validation/255

#%%
k = 8 * 8 * 8
n = 32*32*3
#Make sure we devide k by two in the line below
sqrtk = np.sqrt(k / 2)
c = k // 64
snr = 0
p = 1
var = p / math.pow(10, snr / 10)
var = var/2 #var should be devided by 2
std = np.sqrt(var)
width = 32
height = 32
batch_size = 64
nb_epochs = 15
code_length = 128
print(std, std ** 2, 'k/n: ', k / (2 * n))
#%%

K.clear_session()
tf.set_random_seed(0)
np.random.seed(0)

#from keras.mode import Model
#encoder part
input = Input(shape=(32,32,3))
conv_1 = Conv2D(16,(5,5),padding = 'same', strides = 2,activation='relu')(input)
conv_1 = BatchNormalization()(conv_1)
conv_2 = Conv2D(32,(5,5),padding = 'same', strides = 2,activation='relu')(conv_1)
conv_2 = BatchNormalization()(conv_2)
conv_3 = Conv2D(32,(5,5),padding = 'same', strides = 1,activation='relu')(conv_2)
conv_3 = BatchNormalization()(conv_3)
conv_4 = Conv2D(32,(5,5),padding = 'same', strides = 1,activation='relu')(conv_3)
conv_4 = BatchNormalization()(conv_4)
encoded = Conv2D(c,(5,5),padding = 'same', strides = 1,activation='relu')(conv_4)


z_mean = Conv2D(c,(5,5),padding = 'same', strides = 1,activation='relu')(encoded)
z_log_var = Conv2D(c,(5,5),padding = 'same', strides = 1,activation='relu')(encoded)

#%%


#reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0],K.shape(z_mean)[1],K.shape(z_mean)[2],K.shape(z_mean)[3]), mean=0.,
                              stddev=1.0)
    return z_mean + K.exp(z_log_var / 2) * epsilon


from keras.layers import Input, Dense, Lambda,Reshape, Flatten
z = Lambda(sampling, output_shape=(8,8,c))([z_mean, z_log_var])
z = Flatten()(z)
#%%

class ChannelNormalizer(Layer):

    def __init__(self, sqrtk, **kwargs):
        self.sqrtk = sqrtk
        super(ChannelNormalizer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ChannelNormalizer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return self.sqrtk * K.l2_normalize(x, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape


z = ChannelNormalizer(sqrtk, name='normal')(z)

#%%
class ChannelNoise(Layer):

  def __init__(self, sigma, **kwargs):
    self.sigma = sigma
    super(ChannelNoise, self).__init__(**kwargs)

  def build(self, input_shape):
    self.inshape = input_shape
    super(ChannelNoise, self).build(input_shape)

  def call(self, x):
    return x + K.random_normal(self.inshape[1:], mean = 0, stddev = self.sigma)

  def compute_output_shape(self, input_shape):
    return input_shape

z = ChannelNoise(std)(z)
#%%

#decoder part
z = Reshape([8,8,c])(z)
conv_0T = Conv2DTranspose(32,(5,5), padding = 'same', strides = 1,activation='relu')(z)
conv_0T = BatchNormalization()(conv_0T)
conv_0T = Dropout(0.1)(conv_0T)
conv_1T = Conv2DTranspose(32,(5,5), padding = 'same', strides = 1,activation='relu')(conv_0T)
conv_1T = BatchNormalization()(conv_1T)
conv_2T = Conv2DTranspose(32,(5,5), padding = 'same', strides = 1,activation='relu')(conv_1T)
conv_2T = BatchNormalization()(conv_2T)
conv_2T = Dropout(0.1)(conv_2T)
conv_3T = Conv2DTranspose(16,(5,5), padding = 'same', strides = 2,activation='relu')(conv_2T)
conv_3T = BatchNormalization()(conv_3T)
x_out = Conv2DTranspose(3,(5,5), padding = 'same', strides = 2,activation='sigmoid')(conv_3T)

#%%

vae = Model(input, x_out)

from keras import metrics


def VAE_loss(x_origin,x_out):
    kl_tolerance = k/(32*32*3)
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_origin- x_out), axis=[1, 2, 3]))
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    kl_loss = tf.reduce_mean(tf.maximum(kl_loss, kl_tolerance * (32*32*3)))
    loss_sum = reconstruction_loss + kl_loss
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
chckpnt = keras.callbacks.ModelCheckpoint(save_directory + 'vae_0db_1_12_weights.{epoch}-{val_PSNR:.2f}.h5',
                                                    monitor='val_PSNR', verbose=0, save_best_only=False,
                                                    save_weights_only=True, mode='auto', period=100)
csv = keras.callbacks.CSVLogger(save_directory + 'vae_0db_1_12.log', separator=',', append=True)
opt = keras.optimizers.Adam(lr=0.001)

vae.compile(optimizer=opt, loss=VAE_loss, metrics=[PSNR])
#TODO uncomment line below to load weights

# vae.load_weights()
vae.fit(X_train_norm, X_train_norm, shuffle=True, epochs=5000, batch_size=64,
        validation_data=(X_validation_norm, X_validation_norm), callbacks=[lrate, chckpnt, csv])

for i in range(10):
    print(vae.evaluate(X_test_norm, X_test_norm))