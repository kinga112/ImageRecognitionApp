import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import time


dataDir = "datasets"
categories = ["baseball", "basketball"]

for category in categories:
    path = os.path.join(dataDir, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')
        break
    break

img_size = 64
new_array = cv2.resize(img_array,(img_size, img_size))
plt.imshow(new_array, cmap='gray')

trainingData = []

def createTrainingData():
    for category in categories:
        path = os.path.join(dataDir, category)
        classNum = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                trainingData.append([new_array, classNum])
            except Exception as e:
                pass

createTrainingData()

random.shuffle(trainingData)

X = []
y = []

for features, label in trainingData:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 1) #change 1 to 3 for RGB colors

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

X = X/255.0

dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            model.fit(X, y, batch_size=10, epochs=50, validation_split=0.3)

model.save('baseball-basketball-1.model')