from flask import Flask, render_template, send_file, Response, Blueprint, redirect, url_for
import subprocess
from bicep_curls import infer

app = Flask(__name__)

about_blueprint = Blueprint('about', __name__)
resources_blueprint = Blueprint('resources', __name__)
menst_blueprint = Blueprint('menst', __name__)


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
    infer()
    # Redirect back to the home page after the function completes
    return redirect(url_for('index'))

@app.route('/shoulder')
def shoulder():
    infer()
    # Redirect back to the home page after the function completes
    return redirect(url_for('index'))





if __name__ == '__main__':
    app.run(debug=True)
