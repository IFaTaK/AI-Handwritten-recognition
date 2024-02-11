import os
import subprocess
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

print("Do you wannt to load or to create a new model? (Press Enter to load, type 'new' to create a new model) ")
if input() == 'new':
    name = input("Name of the model: ")
    model = createModel()
    trainModel(model)
    model.save('model/'+ name +'.model')
else:
    models = subprocess.check_output('ls model/ | grep .model', shell=True).decode('utf-8')
    models = models.split('\n')
    models.pop()
    models_list = [model.split('.')[0] for model in models]
    models = " ".join(models_list)

    str_model = ".model"
    while str_model not in models_list:
        print('which model do you want to use?')
        print(models)
        str_model = input()
        if str_model in models_list:
            model = keras.models.load_model('model/' + str_model + '.model')
            break
        else:
            if str_model == "leave":
                exit()
            print("Model not found")
            continue

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