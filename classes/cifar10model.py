from .abstractmodel import AbstractImageClassificationModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10

class AbstractCIFAR10ImageClassificationModel(AbstractImageClassificationModel):

    # Load dataset and split into training and test sets.
    def load_dataset(self):
        (trainX, trainY), (testX, testY) = cifar10.load_data()
        return trainX, trainY, testX, testY

