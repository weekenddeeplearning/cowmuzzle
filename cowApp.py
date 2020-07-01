from cowsMuzzleDNN import *
import cv2
import io
from PIL import Image

from flask import Flask, request
from flask import render_template

import json

app = Flask(__name__, static_folder='./templates/asserts/')

#trainer = cowsMuzzleTrain()
finder = cowFinder()
default_img = r"D:\Documents\Cows_simils\Muzzle Images 2\testing\001.jpg"
default_trainset = r'./Muzzle Images 2/training'


@app.route('/')
def hello():
    return render_template("index.html") #"Hello World!"


#@app.route('/train')
#def train_model():
#    # train_module = cowsMuzzleTrain()
#    return trainer.training(default_trainset)


@app.route('/find', methods=["POST"])
def find_cow():
    # predict_module = cowFinder()
    #print(request.form['img'])

    decoded_data = base64.b64decode(request.form['img'].replace('data:image/jpeg;base64,',''))
    np_data = np.fromstring(decoded_data, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)

    if img is None:
        img = cv2.imread(default_img)

    result = finder.find_top_two_cows(img, default_trainset)
    #print(result)

    return json.dumps(result)

def getData():
    return None

if __name__ == '__main__':
    app.run()
