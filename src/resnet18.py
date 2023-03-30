from keras import layers
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout

from keras.models import Model
from keras.regularizers import l2
from keras import losses
from keras import optimizers
from keras.optimizers import SGD
import tensorflow as tf

from src.metrics import recall_m, precision_m, f1_m
#Weight decay: pénalisation de la loss en norme L2 pour réduire l'over-fitting
#https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab
#https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/
def conv2d_bn(x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
    layer = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(weight_decay)
                   )(x)
    layer = BatchNormalization()(layer)
    return layer


def conv2d_bn_relu(x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
    layer = conv2d_bn(x, filters, kernel_size, weight_decay, strides)
    layer = Activation('tanh')(layer)
    return layer


def ResidualBlock(x, filters, kernel_size, weight_decay, downsample=True):
    if downsample:
        # residual_x = conv2d_bn_relu(x, filters, kernel_size=1, strides=2)
        residual_x = conv2d_bn(x, filters, kernel_size=1, strides=2)
        stride = 2
    else:
        residual_x = x
        stride = 1
    residual = conv2d_bn_relu(x,
                              filters=filters,
                              kernel_size=kernel_size,
                              weight_decay=weight_decay,
                              strides=stride,
                              )
    residual = conv2d_bn(residual,
                         filters=filters,
                         kernel_size=kernel_size,
                         weight_decay=weight_decay,
                         strides=1,
                         )
    out = layers.add([residual_x, residual])
    out = Activation('relu')(out)
    return out


def ResNet18(classes, input_shape, dropout1, dropout2, dropout3, dropout4, dropout5, weight_decay):
    input = Input(shape=input_shape)
    x = input
    # x = conv2d_bn_relu(x, filters=64, kernel_size=(7, 7), weight_decay=weight_decay, strides=(2, 2))
    # x = MaxPool2D(pool_size=(3, 3), strides=(2, 2),  padding='same')(x)
    x = conv2d_bn_relu(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, strides=(1, 1))

    x=Dropout(dropout1)(x)

    # # conv 2
    x = ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    x = ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)

    x=Dropout(dropout2)(x)

    # # conv 3
    x = ResidualBlock(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    x = ResidualBlock(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)

    x=Dropout(dropout3)(x)

    # # conv 4
    x = ResidualBlock(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    x = ResidualBlock(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)

    x=Dropout(dropout4)(x)

    # # conv 5
    x = ResidualBlock(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    x = ResidualBlock(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    x = AveragePooling2D(pool_size=(4, 4), padding='valid')(x)

    x=Dropout(dropout5)(x)

    x = Flatten()(x)

    # Fully connected

    x = Dense(classes, activation='sigmoid')(x)
    model = Model(input, x, name='ResNet18')
    return model

def buildModel(learnRate=0.001, momentum=0.99, dropout1=0.6, dropout2=0.7,
                        dropout3=0.7, dropout4=0.4, dropout5=0.4, weight_decay=1e-4):

    model = ResNet18(1, (224,224,3), dropout1, dropout2, dropout3, dropout4, dropout5, weight_decay)
    model.build(input_shape = (None,224 ,224 ,3))
    opt = SGD(learning_rate=learnRate, momentum = momentum) 
    model.compile(optimizer = opt,loss=tf.keras.losses.binary_crossentropy, metrics=["Accuracy", recall_m, precision_m, f1_m]) 
    
    return model