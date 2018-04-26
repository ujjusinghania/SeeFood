from flask import Flask, render_template, redirect, url_for, Blueprint, request, session
import os
from PIL import Image
from io import BytesIO
import keras
from keras import Sequential
import numpy as np
import requests

app = Flask(__name__)
folder_names = ["Unknown", "Pizza", "French Fries", "Cheesecake", "Burger"]

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/predictFood', methods=['GET', 'POST'])
def predictFood():
    foodURL = request.args.get('foodImageURL')
    imageResponse = requests.get(foodURL)
    foodImage = Image.open(BytesIO(imageResponse.content))
    foodImage = foodImage.resize((200, 200), Image.ANTIALIAS)
    print(foodURL)
    foodNumpyArray = np.zeros((1, 200, 200, 3))
    foodNumpyArray[0, :, :, :] = np.asarray(foodImage, dtype='int32')
    predictionResult = cnnPredict(foodNumpyArray)
    return render_template('result.html', foodURL=foodURL, unkownP=predictionResult[0], burgerP=predictionResult[4], frenchfriesP=predictionResult[2], cheesecakeP=predictionResult[3], pizzaP=predictionResult[1])

def cnnPredict(foodImage):
    keras.backend.clear_session()
    model = Sequential()
    model = keras.models.load_model('reLu_59,080,821_9897_8072')
    predictions = model.predict(foodImage)
    return predictions[0] * 100

app.secret_key = os.urandom(24)
if __name__ == "__main__":
	app.run('127.0.0.1', 5000, debug=True)


