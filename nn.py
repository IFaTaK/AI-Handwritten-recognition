import os
import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2

def loadingData():
    print("Loading data...")
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("Data loaded")
    x_train = keras.utils.normalize(x_train, axis=1)
    x_test = keras.utils.normalize(x_test, axis=1)
    return x_train, y_train, x_test, y_test

def createModel():
    return keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

def trainModel(model):
    x_train, y_train, x_test, y_test = loadingData()
    print("Training model...")
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    print("Model trained")
    model.evaluate(x_test, y_test)


if os.path.exists('model/nn.model'):
    print("Loading model...")
    model = keras.models.load_model('model/nn.model')
    print("Model loaded")  
else:
    print("Creating model: ", end="")
    model = createModel()
    print("Done")
    trainModel(model)
    print("Saving model...")
    model.save('model/nn.model')
    print("Model saved")


number = 0
while os.path.exists(f'handwritten/number{number}-00.png'):
    image = cv2.imread(f'handwritten/number{number}-00.png')[:,:,0]
    image = np.invert(np.array([image]))
    image = keras.utils.normalize(image, axis=1)
    prediction = model.predict(image, verbose=0)
    print(f'Prediction: {np.argmax(prediction)} | Confidence: {np.max(prediction)*100:.2f}%')
    # plt.imshow(image[0], cmap=plt.cm.binary)
    # plt.show()
    number += 1