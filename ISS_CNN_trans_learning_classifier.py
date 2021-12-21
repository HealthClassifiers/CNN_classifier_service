from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot
from tensorflow import keras

from medical_data_utils import MedicalDataUtils

class TransferLearningModel:
    def load_data(self, pref_width = 32, pref_height = 32):
        data_utils = MedicalDataUtils()

        print(" > Loading and preprocssing dataset...")
        self.trainX, self.trainY, self.testX, self.testY = data_utils.load_medical_data(pref_width=pref_width, pref_height=pref_height)

    def load_pre_trained_model(self):
        self.base_model = VGG16(weights="imagenet", include_top=False, input_shape=self.trainX[0].shape)
        self.base_model.trainable = False
        self.base_model.summary()

    def add_fully_connected_layers(self):
        flatten_layer = layers.Flatten()
        dense_layer1 = layers.Dense(50, activation='relu')
        dense_layer2 = layers.Dense(20, activation='relu')
        prediction_layer = layers.Dense(22, activation='softmax')

        self.model = models.Sequential([
            self.base_model,
            flatten_layer,
            dense_layer1,
            dense_layer2,
            prediction_layer
        ])

    def print_model(self):
        self.load_pre_trained_model()
        self.add_fully_connected_layers()
        self.model.summary()

    def compile(self):
        self.model.compile(
            optimizer='adam', 
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5,  restore_best_weights=True)

        history = self.model.fit(self.trainX, self.trainY, epochs=30, validation_split=0.2, batch_size=1, callbacks=[es], verbose=1)
        return history

    # plot diagnostic learning curves
    def summarize_diagnostics(self, filename, history):
        # plot loss
        pyplot.subplot(211)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(history.history['loss'], color='blue', label='train')
        pyplot.plot(history.history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(212)
        pyplot.title('Classification Accuracy')
        pyplot.plot(history.history['accuracy'], color='blue', label='train')
        pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
        # save plot to file
        # filename = sys.argv[0].split('/')[-1]
        pyplot.tight_layout()
        pyplot.savefig(filename + '_plot.png')
        pyplot.close()

    def save_model(self, name):
        #Since we appended cs231
        # sys.path.remove('../cs231n')
        self.model.save('./models/' + name)

    def save_summed_results(self, details_about_model, acc):
        f = open("./CNNs/results/summed_res.txt", "a")
        msg = f"{details_about_model} ,ACCURACY: {acc}"
        f.write(msg)
        f.close()
