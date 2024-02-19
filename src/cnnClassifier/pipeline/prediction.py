import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class Prediction:
    def __init__(self, filename) -> None:
        self.filename = filename
        self.classes = ['Adenocarcinoma','Large cell carcinoma','Normal','Squamous cell carcinoma']
    
    def predict(self):
        model = load_model(os.path.join('model', 'model.h5'))
        test_img = image.load_img(self.filename, target_size=(224, 224))
        test_img = image.img_to_array(test_img)/255.0
        test_img = np.expand_dims(test_img, axis=0)
        result = np.argmax(model.predict(test_img), axis=-1)
        return {'image': self.classes[result[0]]}
    

    