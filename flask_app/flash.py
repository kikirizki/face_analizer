from flask import flask, render_template, Response
from camera import Camera

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')
