import keras.layers as layers
import keras
import tensorflow as tf
from src.metrics import recall_m, precision_m, f1_m, f2_m
from keras.optimizers import SGD

def buildModel(learnRate=0.01, dropout1=0.2, dropout2=0.2, dropout3=0.2, dropout4=0.2, momentum=0.9):
    model = keras.Sequential()
    # Layer 1 Conv2D
    model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(28,28,3), padding="same"))
    # Layer 2 Pooling Layer
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    #Dropout1
    model.add(layers.Dropout(dropout1))

    # Layer 3 Conv2D
    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    # Layer 4 Pooling Layer
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    
    #Dropout2
    model.add(layers.Dropout(dropout2))

    model.add(layers.Flatten())

    # Fully connected
    model.add(layers.Dense(units=120, activation='tanh'))

    #Dropout3
    model.add(layers.Dropout(dropout3))

    model.add(layers.Dense(units=84, activation='tanh'))

    #Dropout4
    model.add(layers.Dropout(dropout4))
    model.add(layers.Dense(units=1, activation='sigmoid'))

    sgd = SGD(learning_rate=learnRate, momentum=momentum)
    model.compile(optimizer=sgd,loss=tf.keras.losses.binary_crossentropy,metrics=["Accuracy", recall_m, precision_m, f1_m, f2_m])

    return model