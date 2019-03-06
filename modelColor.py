import os
import cv2
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

dataDir = "datasets"
categories = ["baseball", "basketball", "volleyball", "soccer ball", "hockey puck"]

imgSize = 128

trainingData = []

def createTrainingData():
    for category in categories:
        path = os.path.join(dataDir, category)
        index = categories.index(category)
        for img in os.listdir(path):
            try:
                imgArray = cv2.imread(os.path.join(path, img))
                newArray = cv2.resize(imgArray, (imgSize, imgSize))
                trainingData.append([newArray, index])
            except Exception as e:
                pass

createTrainingData()

random.shuffle(trainingData)

X = []
y = []

for features, label in trainingData:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, imgSize, imgSize, 3)
y = np.array(y)

X = X/255.0

#input layers
model = Sequential()

model.add(Conv2D(20, (5, 5), padding="same", input_shape=(imgSize, imgSize, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(50, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())

#hidden layer 1
model.add(Dense(128))
model.add(Activation("relu"))

#hidden layer 2
model.add(Dense(128))
model.add(Activation("relu"))

#output layer
model.add(Dense(5))
model.add(Activation("softmax"))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=32)
model.save('models/modelColor/color-model-1.model')