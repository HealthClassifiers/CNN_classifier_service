from io import BytesIO
from PIL import Image
from pymongo import MongoClient
from pymongo.database import Database
from werkzeug.datastructures import FileStorage
import gridfs
class PersistanceModule:
    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017")
        self.db = self.client.db
        self.trainingData = self.db.trainingData

    def insert_user_data(self, img):
      #  fs = gridfs.GridFS(self.db)
      #  with open(img, "rb") as image_file:
      #      encoded_string = image_file.read()
      #  fs.put(encoded_string)
        #self.trainingData.insert_one(img)
        print(img)

    def show_user_data(self):
        images = self.trainingData._list_collections()
        print(images)


    def getDb(self) -> Database:
        return self.db