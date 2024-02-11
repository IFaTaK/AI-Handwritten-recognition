import subprocess
import keras

def modelInfo(str_model):
    print("Loading model...")
    model = keras.models.load_model('model/' + str_model + '.model')
    print("Model loaded")
    model.summary()

models = subprocess.check_output('ls model/ | grep .model', shell=True).decode('utf-8')
models = models.split('\n')
models.pop()
models_list = [model.split('.')[0] for model in models]
models = " ".join(models_list)

str_model = ".model"
while str_model not in models_list:
    print('which model do you want to see?')
    print(models)
    str_model = input()
    print(str_model, models_list)
    if str_model in models_list:
        modelInfo(str_model)
        break
    else:
        print("Model not found")
        continue