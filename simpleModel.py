import os
import cv2
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten

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
                imgArray = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
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

X = np.array(X).reshape(-1, imgSize, imgSize, 1)
y = np.array(y)

X = X/255.0

#input layer
model = Sequential()
model.add(Flatten())

#hidden layer 1
model.add(Dense(128, activation='relu'))

#hidden layer 2
model.add(Dense(128, activation='relu'))

#output layer
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=32)
model.save('models/simpleModel/simple-model-2.model')