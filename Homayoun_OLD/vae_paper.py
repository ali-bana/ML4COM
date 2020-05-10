import numpy as np
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras import backend as K
from keras.layers import Layer, PReLU, Conv2D, Activation, Conv2DTranspose , GaussianNoise,BatchNormalization,Conv1D
from sklearn.model_selection import train_test_split
import tensorflow as tf
import math

# Load dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
# Divide data into test and validdation
X_test, X_validation, Y_test,Y_validation  = train_test_split(X_test, Y_test, test_size=0.33, random_state=42)

# Normalizing dataset
X_train_norm = X_train/255
X_test_norm = X_test/255
X_validation_norm = X_validation/255


k = 8 * 8 * 8
n = 32*32*3
#Make sure we devide k by two in the line below
sqrtk = np.sqrt(k / 2)
c = k // 64
snr = 0
p = 1
var = p / math.pow(10, snr / 10)
var = var/2 #var should be devided by 2
stddev = np.sqrt(var)
np.random.seed(1000)
width = 32
height = 32
batch_size = 64
nb_epochs = 15
code_length = 128
print(stddev, stddev ** 2, 'k/n: ', k / (2 * n))


from keras.layers import Input, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Model
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


#reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0],K.shape(z_mean)[1],K.shape(z_mean)[2],K.shape(z_mean)[3]), mean=0.,
                              stddev=1.0)
    return z_mean + K.exp(z_log_var / 2) * epsilon


from keras.layers import Input, Dense, Lambda,Reshape
z = Lambda(sampling, output_shape=(8,8,c))([z_mean, z_log_var])
z = Flatten()(z)

class ChannelNormalizer(Layer):

  def __init__(self,sqrtk, **kwargs):
#     self.output_dim = output_dim
    self.sqrtk=sqrtk
    super(ChannelNormalizer, self).__init__(**kwargs)

  def build(self, input_shape):
    super(ChannelNormalizer, self).build(input_shape)  # Be sure to call this at the end

  def call(self, x):
    return self.sqrtk*K.l2_normalize(x,axis=1)
  def compute_output_shape(self, input_shape):
    return input_shape

#z = Reshape([8,8,c])(z)
#z = keras.layers.average([z,encoded])

z = ChannelNormalizer(sqrtk)(z)


import math
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
    return x  + K.random_normal(self.inshape[1:], mean = 0, stddev = self.sigma)

  def compute_output_shape(self, input_shape):
    return input_shape

z = ChannelNoise(math.sqrt(0.1))(z)

z = Reshape((8, 8, c))(z)

#decoder part

conv_0T = Conv2DTranspose(32,(5,5), padding = 'same', strides = 1,activation='relu')(z)
conv_0T = BatchNormalization()(conv_0T)
conv_1T = Conv2DTranspose(32,(5,5), padding = 'same', strides = 1,activation='relu')(conv_0T)
conv_1T = BatchNormalization()(conv_1T)
conv_2T = Conv2DTranspose(32,(5,5), padding = 'same', strides = 1,activation='relu')(conv_1T)
conv_2T = BatchNormalization()(conv_2T)
conv_3T = Conv2DTranspose(16,(5,5), padding = 'same', strides = 2,activation='relu')(conv_2T)
conv_3T = BatchNormalization()(conv_3T)
x_out = Conv2DTranspose(3,(5,5), padding = 'same', strides = 2,activation='sigmoid')(conv_3T)

vae = Model(input, x_out)

from keras import metrics
def VAE_loss(x_origin,x_out):
    # x_origin=K.flatten(x_origin)
    # x_out=K.flatten(x_out)
    # xent_loss =  32*32*3*metrics.mean_squared_error(x_origin, x_out)
    # kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    # vae_loss = K.mean(10*xent_loss+kl_loss)
    kl_tolerance = 2
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_origin- x_out), axis=[1, 2, 3]))
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    kl_loss = tf.reduce_mean(tf.maximum(kl_loss, kl_tolerance * (32*32*3)))
    loss_sum = reconstruction_loss + kl_loss

    return loss_sum
def PSNR(y_true,y_pred):

  return 10 * K.log(K.max(y_true)**2 / (K.mean(K.square(y_pred-y_true)))) / K.log(10.0)

opt = keras.optimizers.Adam(lr=0.001)

vae.compile(optimizer=opt, loss=VAE_loss,metrics=[PSNR])

vae.fit(X_train_norm,X_train_norm,shuffle=True,epochs=5000,batch_size=640,validation_data=(X_validation_norm, X_validation_norm))

print(vae.evaluate(X_test_norm, X_test_norm))




from keras.models import Model

normal = Model(inputs=vae.input, outputs=vae.get_layer('normal').output)
noise = Model(inputs=vae.input, outputs=vae.get_layer('noise').output)
normal_out = normal.predict(X_test_norm[:10])
noise_out = noise.predict(X_test_norm[:10])
for i in range(len(normal_out)):
  print(np.sum(np.square(normal_out[i])), k)
  print(noise_out[i] - normal_out[i])
  print('............................')


print('ldjfnvkldsjfbndskjbnsfkgb')

normal_out = normal.predict(X_test_norm[:10])
noise_out = noise.predict(X_test_norm[:10])
for i in range(len(normal_out)):
  print(np.sum(np.square(normal_out[i])), k)
  print(noise_out[i] - normal_out[i])
  print('............................')




