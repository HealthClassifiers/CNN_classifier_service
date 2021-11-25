from pymongo import MongoClient

class PersistanceModule:
    def __init__(self):
        self.client = MongoClient("mongodb://mongo:27017")
        self.db = self.client.user_data
    

    def insert_user_data(self, img):
        #self.db.insert_one(img)
        print(img)

    def show_user_data(self):
        images = self.db._list_collections()
        print(images)
