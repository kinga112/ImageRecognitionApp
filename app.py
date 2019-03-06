import cv2
import tensorflow as tf

categories = ["baseball", "basketball", "volleyball", "soccer ball", "hockey puck", "football"]

def prepare(filepath):
    imgSize = 128
    #imgArray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) #simpleModel.py
    imgArray = cv2.imread(filepath) #modelColor.py
    newArray = cv2.resize(imgArray, (imgSize, imgSize))
    #return newArray.reshape(-1, imgSize, imgSize, 1) #simpleModel.py
    return  newArray.reshape(-1, imgSize, imgSize, 3) #modelColor.py

model = tf.keras.models.load_model("models/modelColor/color-model-3.model")

prediction = model.predict([prepare("predictionImages/baseballimage.jpg")])

for i in range(6):
    if prediction[0][i] > 0.5:
        print(categories[i])

prediction = model.predict([prepare("predictionImages/hockeypuckimage.jpg")])

for i in range(6):
    if prediction[0][i] > 0.5:
        print(categories[i])

prediction = model.predict([prepare("predictionImages/soccerballimage2.jpg")])

for i in range(6):
    if prediction[0][i] > 0.5:
        print(categories[i])

prediction = model.predict([prepare("predictionImages/basketballimage.jpg")])

for i in range(6):
    if prediction[0][i] > 0.5:
        print(categories[i])

prediction = model.predict([prepare("predictionImages/volleyballimage.jpg")])

for i in range(6):
    if prediction[0][i] > 0.5:
        print(categories[i])

prediction = model.predict([prepare("predictionImages/footballimage.jpg")])

for i in range(6):
    if prediction[0][i] > 0.5:
        print(categories[i])