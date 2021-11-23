from logging import error
import numpy as np

from flask import Flask, request, make_response
from finished_classifier import finished_classifier
from user_data_persistance import PersistanceModule

error_messages = {'NO_IMAGE_PROVIDED': 'You have not provided an image in correct format to be classified.'}

app = Flask("ISS_Classifier")

@app.route('/classify', methods=['POST'])
def classify_image():
    file = request.files['image'].read()
    if file != None:
        npimg = np.fromstring(file, np.uint8)
        classifier_service = finished_classifier()
        identified_class = classifier_service.classify_image(npimg)
        return identified_class
    else:
        return error_messages(error_messages['NO_IMAGE_PROVIDED'])

@app.route('/test', methods=['GET'])
def test():
    db = PersistanceModule()
    
    classifier = finished_classifier(save_on_classification=False)

    print(classifier)

    result = classifier.classify_image('./test_img.jpg')

    # db.show_user_data()
    response = make_response(result, 200)
    response.mimetype = "test/plain"
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
