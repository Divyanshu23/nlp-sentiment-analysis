from flask import Flask, render_template, flash, request, url_for
import numpy as np
from numpy.core.defchararray import array
from numpy.lib.type_check import imag
import pandas as pd
import re
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import models

IMAGE_FOLDER = os.path.join("static","images")

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = IMAGE_FOLDER

def init():

  global model
  # Load the pre-trained keras model
  model = models.load_model("/media/divyanshu/My Data/sentiment-analysis/app")


@app.route("/", methods=["GET","POST"])
def home():
  return render_template("home.html")

@app.route("/sentiment_analysis_prediction", methods=["POST", "GET"])
def prediction():
  if request.method=="POST":
    text = request.form["text"]
    sentiment = ""
    max_review_len = 500
    word_to_id = imdb.get_word_index()
    strip_special_chars = re.compile("[A-Za-z0-9]")
    text = text.lower().replace("<br />"," ")
    text = re.sub(strip_special_chars, "", text.lower())

    words = text.split()
    X_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word]<=20000) else 0 for word in words]]
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_len)
    vector = np.array([X_test.flatten()])

    prob = model.predict(np.array(vector[0]))[0][0]
    class_pred = model.predict_classes(np.array(vector[0]))[0][0]

    if class_pred == 0:
      sentiment = "negative"
      img_file = os.path.join(app.config["UPLOAD_FOLDER"],"sad.png")
    else:
      sentiment = "positve"
      img_file = os.path.join(app.config["UPLOAD_FOLDER"],"happy.png")
  return render_template("home.html", text=text, sentiment=sentiment, probability=prob, image=img_file)

if __name__=="__main__":
  init()
  app.run()