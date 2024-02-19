import os
import tensorflow as tf
from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

    def train_valid_generator(self):
        
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest'
        )

        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )

        self.valid_generator = valid_datagen.flow_from_directory(
            self.config.valid_data,
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
        )

        self.train_generator = train_datagen.flow_from_directory(
            self.config.training_data,
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
        )
    

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        os.makedirs('model', exist_ok=True)
        model.save('model/model.h5')
        model.save(path)
    
    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.valid_generator,
            validation_steps=self.validation_steps
        )

        self.save_model(self.config.trained_model_path, self.model)