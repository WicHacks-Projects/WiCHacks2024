from flask import Flask, render_template, send_file, Response, Blueprint, redirect, url_for
import subprocess
import pymongo
from bicep_curls import infer
from lateral_lifts import lift
from shoulder_press import press

app = Flask(__name__)

about_blueprint = Blueprint('about', __name__)
resources_blueprint = Blueprint('resources', __name__)
menst_blueprint = Blueprint('menst', __name__)

try:
    client = pymongo.MongoClient("mongodb+srv://wichacks24:Sw3d5QDVQLncHcSR@empowher.qo6k2bq.mongodb.net/?retryWrites=true&w=majority&appName=empowher")
except pymongo.error.ConfigurationError:
    print('db connection error')


db = client.empowher



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/resources')
def resources():
    return render_template('resources.html')

@app.route('/menst')
def menst():
    return render_template('resources2.html')

@app.route('/bicep_curls')
def video_feed():
    infer()
    # Redirect back to the home page after the function completes
    return redirect(url_for('index'))


@app.route('/lateral_lifts')
def lateral():
    lift()
    return redirect(url_for('index'))

    # Redirect back to the home page after the function completes

@app.route('/shoulder')
def shoulder():
    press()
    # Redirect back to the home page after the function completes
    return redirect(url_for('index'))





if __name__ == '__main__':
    app.run(debug=True)
