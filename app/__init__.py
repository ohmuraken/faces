from os import path

from flask import Flask


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = path.join(path.dirname(__file__), 'uploads')
app.config['SECRET_KEY'] = 'secret_key'

from app import views