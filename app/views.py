from random import getrandbits
from os import path
from datetime import timedelta

from flask import (flash, redirect, render_template, request,
                   session, send_from_directory, url_for)


from app import app
from app.libs.crop_face import get_cropped_face
from app.libs.learn import classify_pca_svm

ALLOWED_EXTENSIONS = {'jpg', 'JPG', 'png'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def get_filepath():
    return path.join(app.config['UPLOAD_FOLDER'], session['id'])


@app.before_request
def session_setting():
    # session.permanent = True
    # app.permanent_session_lifetime = timedelta(minutes=3)
    if not session.get('id'):
        session['id'] = str(getrandbits(64))


@app.route('/')
def index():
    if path.isfile(get_filepath()):
        redirect(url_for('refresh'))
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            file.save(get_filepath())
            return redirect(url_for('index'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/refresh')
def refresh():
    '''For debug.'''
    # session['id'] = None
    session.clear()
    return redirect(url_for('index'))


@app.route('/classify')
def classify():
    if not get_filepath():
        return redirect(url_for('index'))

    # is_cropped = get_cropped_face(get_filepath())
    is_cropped = get_cropped_face("/var/www/7faces/app/uploads/" + session["id"])
    if not is_cropped:
        flash('ERROR:No face detected!!')
        return redirect(url_for('index'))

    filepath = '/uploads/' + session['id']
    # filepath = '/uploads/' + session['id'] + "_cropped.png"
    return render_template('result.html',
                           filepath=filepath,
                           name=classify_pca_svm(is_cropped))
