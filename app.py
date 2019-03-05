import cv2
import tensorflow as tf

categories = ["baseball", "basketball"]

def prepare(filepath):
    imgSize = 64
    imgArray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    newArray = cv2.resize(imgArray, (imgSize, imgSize))
    return newArray.reshape(-1, imgSize, imgSize, 1)

model = tf.keras.models.load_model("baseball-basketball-1.model")

prediction = model.predict([prepare("baseballimage.jpg")])
print(categories[int(prediction[0][0])])

prediction = model.predict([prepare("basketballimage.jpg")])
print(categories[int(prediction[0][0])])