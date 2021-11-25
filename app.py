from logging import error
import numpy as np
from PIL import Image
from flask import Flask, request, make_response
from finished_classifier import finished_classifier
from user_data_persistance import PersistanceModule
from flask_socketio import SocketIO, emit
from io import BytesIO  
from flask_cors import CORS, cross_origin
import socketio
from pymongo import MongoClient
error_messages = {'NO_IMAGE_PROVIDED': 'You have not provided an image in correct format to be classified.'}


app = Flask("ISS_Classifier")
cors = CORS(app)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, logger=True , cors_allowed_origins='*')

@app.route('/classify', methods=['POST'])
def classify_image():
    file = request.files['image']
    if file != None:
        classifier_service = finished_classifier()
        identified_class = classifier_service.classify_image(file)
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
    socketio.emit('result' , img_byte_arr.getvalue() )
    # db.show_user_data()
    response = make_response('Ok', 200)
    response.mimetype = "test/plain"
    return response

@app.route('/medicalRecords', methods=['POST'])
def addMedicalRecord():
    print('received message: ' )
    client = MongoClient("mongodb://mongo:27017")
    db = client.medicalrecords
    db.medicalrecords.insert_one()

@socketio.on('message')
def handle_message(data):
    print('received message: ' + data)


if __name__ == "__main__":
    #app.run(host="0.0.0.0", debug=True)
     socketio.run(app, debug=True)
