#from load import *
# from keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras import Model
# from tensorflow.keras.applications.imagenet_utils import decode_predictions
import uuid
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from flask import Flask, render_template, request, session, redirect, current_app, make_response
from pymongo import MongoClient
from tensorflow.keras.utils import load_img
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image
import cv2
from tensorflow.keras.models import model_from_json
import keras.models
import re
import sys
import os
from skimage import io
from dotenv import load_dotenv
import io
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import base64
from io import BytesIO
from datetime import datetime

sys.path.append(os.path.abspath("./model"))


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField('Upload File')


load_dotenv()


global graph, model

app = Flask(__name__)
app.config['SECRET_KEY'] = uuid.uuid4().hex
app.config['UPLOAD_FOLDER'] = 'static/uploads'


client = MongoClient(os.environ.get("MONGO_URI"))
app.db = client.get_default_database()
users = {}


# @app.route('/', methods=['POST'])
# def home():
#     return render_template('')


@app.route('/sign_in', methods=['GET', 'POST'])
def sign_in():
    is_signed_in = request.cookies.get('isLoggedIn')
    if(is_signed_in):
        return redirect('/')
    print(is_signed_in)
    errors = {}
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = current_app.db.users.find_one(
            {'email': email, 'password': password})

        if(user):
            # return redirect('/')
            resp = make_response(redirect('/'))
            resp.set_cookie('isLoggedIn', value='1', max_age=90 * 60)
            resp.set_cookie('user_id', value=user["_id"], max_age=90 * 60)
            return resp
        else:
            errors['e'] = 1
            return render_template('sign_in.html', errors=errors)

    return render_template('sign_in.html', errors=errors)


@app.route('/sign_up', methods=['GET', 'POST'])
def sign_up():

    is_signed_in = request.cookies.get('isLoggedIn')
    if(is_signed_in):
        return redirect('/')

    errors = {}
    if(request.method == 'POST'):
        #####################################
        first_name = request.form.get('fname')
        last_name = request.form.get('lname')
        email = request.form.get("nnemail")
        password = request.form.get('n_pass')
        confirm_password = request.form.get('c_pass')

        if(len(password) < 6):
            errors["password"] = 1
        if(password != confirm_password):
            errors['password'] = 2

        if(errors):
            return render_template('sign_up.html', errors=errors)

        users[email] = [password]
        if(not errors):
            current_app.db.users.insert_one(
                {'_id': uuid.uuid4().hex, 'email': email, 'password': password, 'first_name': first_name, "last_name": last_name})
            # session['email'] = email
            return redirect('/sign_in')
        # print(users)

    return render_template('sign_up.html', errors=errors)


@app.route('/')
def index():

    is_signed_in = request.cookies.get('isLoggedIn')
    if(not is_signed_in):
        return redirect('/sign_in')
    user_id = request.cookies.get('user_id')
    name_of_user = current_app.db.users.find_one({'_id': user_id})
    name_of_user = name_of_user['first_name']
    return render_template('main.html', nof=name_of_user)


@app.route('/history')
def history():
    is_signed_in = request.cookies.get('isLoggedIn')
    if(not is_signed_in):
        return redirect('/sign_in')
    user_id = request.cookies.get('user_id')
    name_of_user = current_app.db.users.find_one({'_id': user_id})
    name_of_user = name_of_user['first_name']
    data = current_app.db.entries.find_one({'_id': user_id})
    if data:
        items = data['items']
    else:
        items = ''
    return render_template('history.html', items=items, nof=name_of_user)


# Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'tif', 'TIF'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

# Function to load and prepare the image in right shape


# def read_image(filename):
#     img = load_img(filename, target_size=(256, 256))
#     # x = img_to_array(img)
#     x = Image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     return x


##### FIRST MODEL IMPLEMENTATION #########
# Load json model
json_file = open('models/resnet-50-MRI.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)


# load weights
# load weights into new model
model.load_weights("models/weights.hdf5")
print("Loaded Model from disk")


# compile and evaluate loaded model
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])


##### Second MODEL IMPLEMENTATION #########
json_file2 = open('models/ResUNet-MRI.json', 'r')
loaded_model_json2 = json_file2.read()
json_file2.close()
model2 = model_from_json(loaded_model_json2)
# load weights
# load weights into new model
model2.load_weights("models/weights_seg.hdf5")
print("Model 2 Loaded from disk ")
# compile and evaluate loaded model
model2.compile(optimizer='adam', loss='focal_tversky', metrics=['tversky'])

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file and allowed_file(file.filename):  # Checking file format
#             filename = file.filename
#             file_path = os.path.join('static\images', filename)
#             print(file_path)
#             # file.save(file_path)
#             img = read_image(file_path)  # prepressing method
#             class_prediction = model.predict(img)
#             print(class_prediction)
#             classes_x = np.argmax(class_prediction)
#             print(classes_x)
#             if classes_x == 0:
#                 fruit = "No Cancer"
#             elif classes_x == 1:
#                 fruit = "Cancer"
#             else:
#                 fruit = "Orange"
#             return render_template('predict.html', fruit=fruit, prob=class_prediction, user_image=file_path)
#         else:
#             return "Unable to read the file. Please check file"


# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file and allowed_file(file.filename):  # Checking file format
#             filename = file.filename
#             file_path = os.path.join('static/images/', filename)
#             print(file_path)
#             # file.save(file_path)
#             img = io.imread(file_path)
#             img = img * 1./255.

#             # Reshaping the image
#             img = cv2.resize(img, (256, 256))

#             # Converting the image into array
#             img = np.array(img, dtype=np.float64)
#             print(img.shape)

#             img = np.reshape(img, (1, 256, 256, 3))

#             class_prediction = model.predict(img)
#             print(class_prediction)
#             classes_x = np.argmax(class_prediction)

#             if classes_x == 0:
#                 fruit = "No Cancer"
#             elif classes_x == 1:
#                 fruit = "Cancer"

#                 # Creating a empty array of shape 1,256,256,1
#                 # X = np.empty((1, 256, 256, 3))

#                 # resizing the image and coverting them to array of type float64
#                 # img = cv2.resize(img, (256, 256))
#                 # img = np.array(img, dtype='float64')

#                 # standardising the image
#                 # img -= img.mean()
#                 # img /= img.std()

#                 # converting the shape of image from 256,256,3 to 1,256,256,3
#                 # X[0, ] = img

#                 # make prediction
#                 # mask = model2.predict(img)

#             else:
#                 fruit = "Orange"

#             print(file_path)
#             return render_template('predict.html', fruit=fruit, prob=class_prediction, user_image=file_path)
#         else:
#             return "Unable to read the file. Please check file"


###########################################################################

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     is_signed_in = request.cookies.get('isLoggedIn')
#     if(not is_signed_in):
#         return redirect('/sign_in')
#     if request.method == 'POST':
#         user_id = request.cookies.get('user_id')
#         ########################################
#         file = request.files['file']
#         filename = file.filename
#         file_path = os.path.join('static/images', filename)
#         # test_image = io.imread(file_path)
#         test_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
#         # test_image = test_image * 1./255.
#         test_image = test_image / 255
#         test_image = cv2.resize(test_image, (256, 256))
#         # img_array = img_to_array(test_image)
#         img_array = np.array(test_image, dtype=np.float64)
#         # img_array = np.reshape(img_array, (256, 256, 3))

#         try:
#             img_batch = np.reshape(img_array, (1, 256, 256, 3))
#         except:
#             img_batc = np.expand_dims(img_array, axis=0)
#             img_batch = preprocess_input(img_batc)

#         res = model.predict(img_batch)

#         if np.argmax(res) == 0:
#             print('no cancer')
#         else:
#             print('cancer')
#             ####################
#             # extract the Mask
#             ####################
#             img = cv2.imread(file_path)
#             X = np.empty((1, 256, 256, 3))
#             img = cv2.resize(img, (256, 256))
#             img = np.array(img, dtype=np.float64)
#             img -= img.mean()
#             img /= img.std()
#             X[0, ] = img
#             predict = model2.predict(X)
#             # Y = np.empty((256,256,256,1)) # one dimension
#             Y = np.empty((256, 256, 256, 3))   # Three Dimesions
#             Y[0, ] = predict
#             mask = Y[0]
#             ####################################################
#             # combine the mask with the image
#             imgwm = cv2.imread(file_path)
#             imgwm = cv2.cvtColor(imgwm, cv2.COLOR_BGR2RGB)
#             imgwm[mask == [1, 1, 1]] = 255

#         # print(decode_predictions(res, top=2)[0])
#     return render_template('predict.html')


@app.route('/logout')
def logout():
    resp = make_response(redirect('/sign_in'))
    resp.delete_cookie('isLoggedIn')
    resp.delete_cookie('user_id')
    return resp


@app.route('/scan', methods=["GET", "POST"])
def scan():

    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data

        path = os.path.join(os.path.abspath(
            os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(file.filename))

        file.save(path)

    is_signed_in = request.cookies.get('isLoggedIn')
    if(not is_signed_in):
        return redirect('/sign_in')

    user_id = request.cookies.get('user_id')
    name_of_user = current_app.db.users.find_one({'_id': user_id})
    name_of_user = name_of_user['first_name']

    if(request.method == 'POST'):
        is_signed_in = request.cookies.get('isLoggedIn')
        if(not is_signed_in):
            return redirect('/sign_in')
        if request.method == 'POST':
            c_date = datetime.today().strftime('%Y-%m-%d')
            user_id = request.cookies.get('user_id')
            # user = current_app.db.users.find_one({'_id': cookie})
            f_name = request.form.get('first_name')
            l_name = request.form.get('last_name')
            #####################
            print(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
            # file_path = os.path.join('static/images', "TCGA_HT_7884_19980913_9.tif")
            im = Image.open(path)
            image_bytes = io.BytesIO()
            im.save(image_bytes, format='JPEG')
            imby = base64.b64encode(
                image_bytes.getvalue()).decode("utf-8")

        ######################
        # print(user)
        # entrie_id = user['entrie_id']
            ########################################
            # test_image = io.imread(file_path)
            test_image = cv2.imread(path, cv2.IMREAD_COLOR)
            # test_image = test_image * 1./255.
            test_image = test_image / 255
            test_image = cv2.resize(test_image, (256, 256))
            # img_array = img_to_array(test_image)
            img_array = np.array(test_image, dtype=np.float64)
            # img_array = np.reshape(img_array, (256, 256, 3))

            try:
                img_batch = np.reshape(img_array, (1, 256, 256, 3))
            except:
                img_batc = np.expand_dims(img_array, axis=0)
                img_batch = preprocess_input(img_batc)

            res = model.predict(img_batch)

            if np.argmax(res) == 0:
                found = False
                defected = 'Not Defected'
                print('no cancer')
            else:
                found = True
                defected = 'Defected'
                print('cancer')
                ####################
                # extract the Mask
                ####################
                img = cv2.imread(path)
                X = np.empty((1, 256, 256, 3))
                img = cv2.resize(img, (256, 256))
                img = np.array(img, dtype=np.float64)
                img -= img.mean()
                img /= img.std()
                X[0, ] = img
                predict = model2.predict(X)
                # Y = np.empty((256,256,256,1)) # one dimension
                Y = np.empty((256, 256, 256, 3))   # Three Dimesions
                Y[0, ] = predict
                mask = Y[0]
                ####################################################
                # combine the mask with the image
                imgwm = cv2.imread(path)
                imgwm = cv2.cvtColor(imgwm, cv2.COLOR_BGR2RGB)
                imgwm[mask == [1, 1, 1]] = 255

                pil_img = Image.fromarray(imgwm)
                buff = BytesIO()
                pil_img.save(buff, format="JPEG")
                new_image_string = base64.b64encode(
                    buff.getvalue()).decode("utf-8")

                # imwm = Image.fromarray(imgwm)
                # mask_bytes = io.BytesIO()
                # imwm.save(mask_bytes, format='JPEG')

        if(current_app.db.entries.find_one({'_id': user_id})):
            # pass
            if found:
                current_app.db.entries.update_one({'_id': user_id}, {"$push": {'items': {'_id': uuid.uuid4(
                ).hex, 'first_name': f_name, 'last_name': l_name, 'image': imby, 'mask': new_image_string, "date": c_date, 'defected': defected}}})
                return render_template('/scan.html', form=form, image=new_image_string, cancer=np.argmax(res), name=f_name, nof=name_of_user)
            else:
                current_app.db.entries.update_one({'_id': user_id}, {"$push": {'items': {'_id': uuid.uuid4(
                ).hex, 'first_name': f_name, 'last_name': l_name, 'image': imby, 'mask': '', "date": c_date, 'defected': defected}}})
                return render_template('/scan.html', form=form, image=imby, cancer=np.argmax(res), name=f_name, nof=name_of_user)

            # print(image_bytes.getvalue())
        else:
            if found:
                current_app.db.entries.insert_one({'_id': user_id, 'items': [{'_id': uuid.uuid4(
                ).hex, 'first_name': f_name, 'last_name': l_name, 'image': imby, 'mask': new_image_string, "date": c_date, 'defected': defected}]})
                return render_template('/scan.html', form=form, image=new_image_string, cancer=np.argmax(res), name=f_name, nof=name_of_user)
            else:
                current_app.db.entries.insert_one({'_id': user_id, 'items': [{'_id': uuid.uuid4(
                ).hex, 'first_name': f_name, 'last_name': l_name, 'image': imby, 'mask': '', "date": c_date, 'defected': defected}]})
                return render_template('/scan.html', form=form, image=imby, cancer=np.argmax(res), name=f_name, nof=name_of_user)

            # print(image_bytes.getvalue())

    return render_template('/scan.html', form=form, nof=name_of_user)
