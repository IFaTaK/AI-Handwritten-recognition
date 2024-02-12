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
    print("Creating model...")
    print("How many hidden layers do you want?")
    hidden_layers = int(input())
    nodes = []
    for i in range(hidden_layers):
        print(f"How many nodes do you want in hidden layer {i+1}?")
        nodes.append(int(input()))
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    for i in range(hidden_layers):
        model.add(keras.layers.Dense(nodes[i], activation='relu'))
        model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(10, activation='softmax'))
    nodes = [28*28] + nodes + [10]
    return model, sum([nodes[i]*nodes[i+1] for i in range(len(nodes)-1)])

def trainModel(model):
    x_train, y_train, x_test, y_test = loadingData()
    print("Training model...")
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    print("Model trained")
    model.evaluate(x_test, y_test)

print("Do you wannt to load or to create a new model? (Press Enter to load, type 'new' to create a new model) ")
if input() == 'new':
    model, size = createModel()
    trainModel(model)
    name = input(f"Name of the model (size: {size}): ")
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