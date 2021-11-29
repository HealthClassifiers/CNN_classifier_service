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
from bson.objectid import ObjectId
error_messages = {'NO_IMAGE_PROVIDED': 'You have not provided an image in correct format to be classified.'}

app = Flask("ISS_Classifier")
app.config['SECRET_KEY'] = 'secret!'

#mongo = pymongo(app)
socket = SocketIO(app, cors_allowed_origins="*" , engineio_logger=True)
CORS(app)
@app.route('/classify', methods=['POST'])
@cross_origin()
def classify_image():
    file = request.files['image']
    if file != None:
        db = PersistanceModule()

        fileBytes =  Binary(file.read())

        classifier_service = finished_classifier()
        identified_class = classifier_service.classify_image(file)

        description = "This is a description"
        comment = ""
        status = "classified"

        encoded = base64.b64encode(fileBytes)
        id = db.getDb().medicalrecords.insert({'data': encoded, 'identified_class':identified_class, 'confidence':42, 'description':description, 'comment':comment, 'status':status})

        result = {'id':str(id), 'identified_class':identified_class, 'confidence':42, 'description':description, 'comment':comment, 'status':status}
        return jsonify(result)
    else:
        return error_messages(error_messages['NO_IMAGE_PROVIDED'])

@app.route('/test', methods=['GET'])
def test():
    db = PersistanceModule()
    
    classifier = finished_classifier(save_on_classification=False)
    result = classifier.classify_image('./test_img.jpg')

    db.show_user_data()
    return make_response(result, 200)

@app.route('/doctor', methods=['POST'])
@cross_origin()
def ping():
    fileId = request.values['fileId']

    db = PersistanceModule()
    record = db.getDb().medicalrecords.find_one({"_id": ObjectId(fileId)})

    db = PersistanceModule()
    db.getDb().medicalrecords.find_one_and_update({"_id": ObjectId(fileId)}, {"$set":{"status":"sent to doctor"}})

    result = {
        'image':record.get('data').decode("utf-8"),
        'id':str(record.get('_id')), 
        'identified_class':record.get('identified_class'), 
        'confidence':record.get('confidence'), 
        'description':record.get('description'), 
        'comment':record.get('comment'), 
        'status':record.get('status')}

    socket.emit('result' , result )

    return make_response("Ok", 200)

@app.route('/medicalRecords', methods=['POST' ])
@cross_origin()
def addMedicalRecord():
    print(request.values)
    fileId = request.values['fileId']
    comment = request.values['comment']
    status = request.values['status']

    db = PersistanceModule()
    s = db.getDb().medicalrecords.find_one_and_update({"_id": ObjectId(fileId)}, {"$set":{"comment":comment, "status":status}})
    print(s)

    return make_response("Ok", 200)

@app.route('/medicalRecords', methods=['GET' ])
@cross_origin()
def getMedicalRecords():
    db = PersistanceModule()

    res = []
    for x in db.getDb().medicalrecords.find():
        res.append({'id':str(x.get('_id')), 'identified_class':x.get('identified_class'), 'confidence':x.get('confidence'), 'description':x.get('description'), 'comment':x.get('comment'), 'status':x.get('status')})
        
    response = make_response(jsonify(res), 200)
    return response

@app.route('/image', methods=['GET' ])
@cross_origin()
def getMedicalRecordImage():
    fileId = request.values['fileId']

    db = PersistanceModule()
    file = db.getDb().medicalrecords.find_one({"_id": ObjectId(fileId)})

    response = make_response(jsonify(file.get('data').decode("utf-8")), 200)
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
