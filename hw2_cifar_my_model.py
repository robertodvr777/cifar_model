import numpy as np
import tensorflow as tf
from keras.utils import np_utils

# You can build your model for the last part of the homework here or elsewhere.
# The method below can be used to generate the required csv file.

def save_csv(x, filename="submission.csv"):
    """save_csv Save the input into csv file

    Arguments:
        x {np.ndarray} -- input array

    Keyword Arguments:
        filename {str} -- The file name (default: {"submission.csv"})

    Raises:
        ValueError: Input data structure is not np.ndarray
    """
    if isinstance(x, np.ndarray):
        x = x.flatten()
        np.savetxt(filename, x, delimiter=',')
    else:
        raise ValueError("The input is not an np.ndarray")


def build_model():
    input_layer = tf.keras.Input(shape=(32,32,3))
    convolution_layer1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(input_layer)
    convolution_layer2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(convolution_layer1)
    maxpooling_layer1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(convolution_layer2)
    dropout_layer1 = tf.keras.layers.Dropout(0.2)(maxpooling_layer1)
    convolution_layer3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(dropout_layer1)
    convolution_layer4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(convolution_layer3)
    maxpooling_layer2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(convolution_layer4)
    dropout_layer2 = tf.keras.layers.Dropout(0.2)(maxpooling_layer2)
    convolution_layer5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(dropout_layer2)
    convolution_layer6 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(convolution_layer5)
    maxpooling_layer3 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(convolution_layer6)
    dropout_layer3 = tf.keras.layers.Dropout(0.2)(maxpooling_layer3)
    flattened_layer1 = tf.keras.layers.Flatten(data_format=None)(dropout_layer3)
    fully_connected_layer1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flattened_layer1)
    dropout_layer4 = tf.keras.layers.Dropout(0.2)(fully_connected_layer1)
    output_layer = tf.keras.layers.Dense(10, activation='softmax')(dropout_layer4)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    return model

def train_and_evaluate(model,Xtrain,Ytrain,Xtest,Ytest):
    optimizer = tf.keras.optimizers.SGD(lr=0.005, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Xtrain, Ytrain, epochs=100, batch_size=64, verbose=2, validation_data=(Xtest, Ytest))
    results = model.evaluate(Xtest, Ytest, verbose=0, batch_size=64)

    return results[1]




def main():
    X = np.load('val_X.npy')
    Y = np.load('val_y.npy')
    test_X = np.load('public_test_X.npy')
    test_Y = np.load('public_test_y.npy')
    one_hot_ytrain = np_utils.to_categorical(Y, 10)
    one_hot_ytest = np_utils.to_categorical(test_Y, 10)
    train_normalized = X.astype('float32')/255.0
    test_normalized = test_X.astype('float32')/255.0
    model = build_model()
    accuracy = train_and_evaluate(model,train_normalized,one_hot_ytrain, test_normalized, one_hot_ytest)
    print(accuracy)


if __name__=="__main__":main()
