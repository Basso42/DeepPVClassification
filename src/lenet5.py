import keras.layers as layers
import keras
import tensorflow as tf
from src.metrics import recall_m, precision_m, f1_m

def buildModel(learnRate=0.01, dropout=0.2):
    model = keras.Sequential()
    # Layer 1 Conv2D
    model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(28,28,3), padding="same"))
    # Layer 2 Pooling Layer
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # Layer 3 Conv2D
    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    # Layer 4 Pooling Layer
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=120, activation='tanh'))
    model.add(layers.Dense(units=84, activation='tanh'))
    model.add(layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='sgd',loss=tf.keras.losses.binary_crossentropy,metrics=["Accuracy", recall_m, precision_m, f1_m])


    #model.compile(loss=loss_function_used, optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), metrics=[keras.metrics.Accuracy(name="Accuracy"),f1_m ,keras.metrics.Precision(name="Precision"), keras.metrics.Recall(name="Recall")])

    return model