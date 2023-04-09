from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
import pickle
import imutils
import sklearn
from tensorflow.keras.models import load_model
# from pushbullet import PushBullet
import joblib
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input


# Loading Models

diabetes_model = pickle.load(open('models/diabetes.sav', 'rb'))


# Configuring Flask
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS



########################### Routing Functions ########################################

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')





########################### Result Functions ########################################





@app.route('/resultd', methods=['POST'])
def resultd():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        pregnancies = request.form['pregnancies']
        glucose = request.form['glucose']
        bloodpressure = request.form['bloodpressure']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        diabetespedigree = request.form['diabetespedigree']
        age = request.form['age']
        skinthickness = request.form['skin']
        pred = diabetes_model.predict(
            [[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age]])
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resultd.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)





# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run(debug=True)
