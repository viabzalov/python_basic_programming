import os
import numpy as np
import cv2 as cv
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)


def colorize(filename):
    # Specify the paths for the 2 model files
    proto_file = './models/colorization_deploy_v2.prototxt'
    weights_file = './models/colorization_release_v2.caffemodel'

    # Read the input image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    frame = cv.imread(image_path)

    # Load the cluster centers
    pts_in_hull = np.load('./pts_in_hull.npy')

    # Read the network into Memory
    net = cv.dnn.readNetFromCaffe(proto_file, weights_file)

    # Populate cluster centers as 1x1 convolution kernel
    pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId('class8_ab')).blobs = [
        pts_in_hull.astype(np.float32)]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [
        np.full([1, 313], 2.606, np.float32)]

    # From opencv sample
    W_in = 224
    H_in = 224

    img_rgb = (frame[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)
    img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
    img_l = img_lab[:, :, 0]  # pull out L channel

    # Resize lightness channel to network input size
    img_l_rs = cv.resize(img_l, (W_in, H_in))
    img_l_rs -= 50  # Subtract 50 for mean-centering

    net.setInput(cv.dnn.blobFromImage(img_l_rs))
    ab_dec = net.forward()[0, :, :, :].transpose(
        (1, 2, 0))  # This is our result

    (H_orig, W_orig) = img_rgb.shape[:2]  # Original image size
    ab_dec_us = cv.resize(ab_dec, (W_orig, H_orig))
    # Concatenate with original image L
    img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)
    img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)

    # Save image
    cv.imwrite(image_path, img_bgr_out * 255)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            colorize(filename)
            return redirect(url_for('uploaded_file', filename=filename))
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <link rel="stylesheet"
              href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"
              integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T"
              crossorigin="anonymous">
        <title>Upload new File</title>
    </head>
    <body>
    <div class="container mt-5">
        <form id="form" action="" method=post enctype=multipart/form-data>
            <p><input type=file name=file></p>
            <button type="submit" class="btn btn-primary">Upload</button>
        </form>
    </div>
    </body>
    </html>
    '''


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_uploads():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)


def create_models():
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)
        os.system('wget https://github.com/richzhang/colorization/blob/master/colorization/resources/pts_in_hull.npy?raw=true -O ./pts_in_hull.npy')
        os.system('wget https://raw.githubusercontent.com/richzhang/colorization/master/colorization/models/colorization_deploy_v2.prototxt -O ./models/colorization_deploy_v2.prototxt')
        os.system('wget http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel -O ./models/colorization_release_v2.caffemodel')


if __name__ == '__main__':
    create_uploads()
    create_models()
    app.run()
