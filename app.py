import cv2
import tensorflow as tf
import os

global hockeyPuck
hockeyPuck = 0
global baseball
baseball = 0
global basketball
basketball = 0

def imageRecog():
    global hockeyPuck
    hockeyPuck = 0
    global baseball
    baseball = 0
    global basketball
    basketball = 0
    #categories = ["baseball", "basketball", "volleyball", "soccer ball", "hockey puck", "football", "tennis ball", "golf ball"]
    categories = ["baseball", "basketball", "hockey puck"]
    def prepare(filepath):
        imgSize = 128
        #imgArray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) #simpleModel.py
        imgArray = cv2.imread(filepath) #modelColor.py
        newArray = cv2.resize(imgArray, (imgSize, imgSize))
        #return newArray.reshape(-1, imgSize, imgSize, 1) #simpleModel.py
        return  newArray.reshape(-1, imgSize, imgSize, 3) #modelColor.py

    model = tf.keras.models.load_model("models/modelColor/color-model-15.model")
    prediction = model.predict([prepare('image1.png')])
    print(prediction)

    for i in range(3):
        if prediction[0][i] > 0.5:
            print(categories[i])
            if categories[i] == "hockey puck":
                global hockeyPuck
                hockeyPuck = 1
            if categories[i] == "baseball":
                global baseball
                baseball = 1
            if categories[i] == "basketball":
                global basketball
                basketball = 1
    os.remove('image1.png')