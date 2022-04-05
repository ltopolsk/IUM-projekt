import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


class CitiesVoc():

    def __init__(self, filename="models/cities_voc.pkl"):
        self.filename = filename
        self.dictonary = self._read_from_file()

    def _read_from_file(self) -> dict:
        with open(self.filename, 'rb') as handle:
            dictonary = pickle.load(handle)
        return dictonary

    def get_dictonary(self):
        return self.dictonary


class CategoryOnehot():

    def __init__(self, filename="models/category_onehot.pkl"):
        self.filename = filename
        self.encoder = self._read_from_file()

    def _read_from_file(self):
        with open(self.filename, 'rb') as handle:
            model = pickle.load(handle)
        return model

    def encode(self, dataframe, col_names):
        return self.encoder.transform(dataframe[col_names])


class StandardScaler():

    def __init__(self, filename="models/standard_scaler.pkl"):
        self.filename = filename
        self.scaler = self._read_from_file()

    def _read_from_file(self):
        with open(self.filename, 'rb') as handle:
            model = pickle.load(handle)
        return model

    def transform_data(self, data):
        return self.scaler.transform(data)


class Classifier():

    def __init__(self, filename):
        self.filename = filename
        self.classifier = self._read_from_file()

    def _read_from_file(self):
        with open(self.filename, 'rb') as handle:
            model = pickle.load(handle)
        return model

    def predict(self, data):
        return self.classifier.predict(data)
