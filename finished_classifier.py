from PIL import Image
from tensorflow import keras
from user_data_persistance import PersistanceModule
import numpy as np
from tensorflow.keras.utils import to_categorical

class finished_classifier:
    def __init__(self, save_on_classification=True, model_version=None) -> None:
        self.save_on_classification = save_on_classification
        self.model = None
        self.db = None
        self.classes = ['Acne and Rosacea', 'Actinic Keratosis Basal Cell Carcinoma', 
                        'Atopic Dermatitis', 'Bullous Disease', 'Cellulitis Impetigo', 
                        'Eczema', 'Exanthems', 'Alopecia', 'Herpes HPV', 
                        'Pigmentation Light Disease', 'Lupus', 'Melanoma', 
                        'Nail Fungus', 'Poison Ivy', 'Psoriasis', 'Scabies Lyme Disease', 
                        'Seborrheic Keratoses (benign tumor)', 'Systemic Disease', 
                        'Tinea Ringworm Candidiasis', 'Urticaria Hives', 
                        'Vascular Tumors', 'Vasculitis', 'Warts Molluscum']

        path_to_prefered_model = 'keras_models/ISS_one_VGG_epoch100_batch64'
        self.model_version = model_version if model_version is not None else path_to_prefered_model


    def load_classifier(self) -> None:
        self.model = keras.models.load_model(self.model_version)

    # Don't uncomment if run through compose - don't want to expand requirements further
    # def visualize_model(self) -> None:
    #     # from keras_visualizer import visualizer 
    #     from tensorflow.keras.models import Sequential
    #     from tensorflow.keras.layers import Conv2D
    #     from tensorflow.keras.layers import MaxPooling2D
    #     from tensorflow.keras.layers import Dense
    #     from tensorflow.keras.layers import Flatten
    #     # from tensorflow.keras.utils import plot_model
    #     from ann_visualizer.visualize import ann_viz

    #     model = Sequential()
    #     model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    #     model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    #     model.add(MaxPooling2D((2, 2)))
    #     model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    #     model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    #     model.add(MaxPooling2D((2, 2)))
    #     model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    #     model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    #     model.add(MaxPooling2D((2, 2)))
    #     model.add(Flatten())
    #     model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    #     model.add(Dense(22, activation='softmax'))

    #     model.summary()

    #     ann_viz(model)     

    def prepare_img(self, img):
        #Make img conform to training data
        img = Image.open(img)
        rezised_img = img.resize((32, 32))

        #Transform to legal numpy array
        data = np.asarray(rezised_img)
        # extra_dim = []
        # extra_dim.append(data)
        data = np.expand_dims(data, axis=0)
        # un_norm_data = np.array(data)
        data = data.astype('float32')
        normalized_data = data / 255.0
        return normalized_data

    def classify_image(self, img) -> int:
        if self.model is None:
            self.load_classifier()

        normalized_data = self.prepare_img(img)

        num_class = self.model.predict(normalized_data)

        y_class = num_class.argmax(axis=-1)[0]

        if self.save_on_classification:
            self.save_data(img)

        return self.classes[y_class]


    def save_data(self, img) -> None: 
        if self.db is None:
            self.db = PersistanceModule()

        self.db.insert_user_data(img)


def test_harness():
    classifier = finished_classifier()
    # classifier.classify_image('./test_img.jpg')
    # classifier.visualize_model()


if __name__ == '__main__': 
    test_harness()