from logging import error
import gridfs
import numpy as np
from json import dumps
from PIL import Image
from flask import Flask, request, make_response , jsonify
import pymongo
from werkzeug.datastructures import FileStorage
from finished_classifier import finished_classifier
from user_data_persistance import PersistanceModule
from flask_socketio import SocketIO, emit
from io import BytesIO  
from flask_cors import CORS, cross_origin
import base64
from bson import Binary
error_messages = {'NO_IMAGE_PROVIDED': 'You have not provided an image in correct format to be classified.'}


app = Flask("ISS_Classifier")
app.config['SECRET_KEY'] = 'secret!'

#mongo = pymongo(app)
socket = SocketIO(app, cors_allowed_origins="*" , engineio_logger=True)
CORS(app)
@app.route('/classify', methods=['POST'])
def classify_image():
    file = request.files['image']
    if file != None:
        db = PersistanceModule()
      #  db.insert_user_data("hej")
      #  with open(file , 'rb') as f:
      #      contents = f.read()
        db.getDb().s.insert({'data': Binary(file.read())})
        

        classifier_service = finished_classifier()
        identified_class = classifier_service.classify_image(file)
        print(identified_class)
        return identified_class
    else:
        return error_messages(error_messages['NO_IMAGE_PROVIDED'])

@app.route('/test', methods=['GET'])
def test():
    db = PersistanceModule()
    
    classifier = finished_classifier(save_on_classification=False)
    result = classifier.classify_image('./test_img.jpg')

    # db.show_user_data()
    response = make_response(result, 200)
    response.mimetype = "test/plain"
    return response

@app.route('/doctor', methods=['POST'])
def ping():
    file = request.files['image']
    img = Image.open(file)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='jpeg')
    socket.emit('result' , img_byte_arr.getvalue() )
    # db.show_user_data()
    response = make_response('Ok', 200)
    response.mimetype = "test/plain"
    return response

@app.route('/medicalRecords', methods=['POST' ])
@cross_origin()
def addMedicalRecord():
    print('received message: ')
    print(request.form)
    file = request.files['blob']
    print(file)
    
    db = PersistanceModule()
      #  db.insert_user_data("hej")
      #  with open(file , 'rb') as f:
      #      contents = f.read()
    res = Binary(file.read())
    encoded = base64.b64encode(res)
    img = 'data:image/png;base64,{}'.format(encoded)
    db.getDb().medicalrecords.insert({'data': encoded})
    
   # img = request.form['blob']
   # print(type(img))
    response = make_response('Ok', 200)
    response.mimetype = "test/plain"
    return response


@app.route('/medicalRecords', methods=['GET' ])
@cross_origin()
def getMedicalRecords():
   
    
    db = PersistanceModule()
      #  db.insert_user_data("hej")
      #  with open(file , 'rb') as f:
      #      contents = f.read()
    res = []
    for x in db.getDb().medicalrecords.find():
        mystr_encoded = base64.b64decode(x.get('data'))
        res.append(x.get('data').decode("utf-8"))
    print(res[1])

       # img = request.form['blob']
   # print(type(img))
    response = make_response(jsonify(res[0]), 200)
    return response



@socket.on('message')
def handle_message(data):
    print('received message: ' + data)

@socket.on('connect')
def connected():
    print('Connected')

if __name__ == "__main__":
    #app.run(host="0.0.0.0", debug=True)
     socket.run(app, debug=True)
