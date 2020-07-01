import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
import base64
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export


class cowsMuzzleTrain():
    def __init__(self):
        print('here')
        self.model = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet', input_shape=(224, 224, 3),
            pooling=None)

    def image_to_embedding(self, image):
        # image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (224, 224))
        img = image[..., ::-1]
        img = np.around(np.transpose(img, (0, 1, 2)) / 255.0, decimals=12)
        x_train = np.array([img])
        embedding = self.model.predict_on_batch(x_train)
        return embedding

    def extract_features(self, path):
        feature_map = {}
        for file in os.listdir(path):
            img = cv2.imread(os.path.join(path, file))
            result = self.image_to_embedding(img)
            # print(result.shape)
            feature_map[file] = result
        return feature_map

    def save_features(self, features):
        pickle_out = open("feature_map.pickle", "wb")
        pickle.dump(features, pickle_out)
        pickle_out.close()

    def training(self, trainset_path):
        feature_map = self.extract_features(trainset_path)
        self.save_features(feature_map)
        return { "status": "completed" }


class cowFinder():
    def loadFeature(self, path):
        pickle_in = open(path, "rb")
        feature_Map = pickle.load(pickle_in)
        return feature_Map

    def __init__(self):
        self.feature_Path = "feature_map.pickle"
        self.feature_Map = self.loadFeature(self.feature_Path)
        self.feature_extractor = cowsMuzzleTrain()

    def get_image(self, closet, trainset):
        with open(os.path.join(trainset,closet), "rb") as img_file:
            text_bytes = base64.b64encode(img_file.read())
            base64_string = text_bytes.decode('utf-8')
        return base64_string

    def find_top_twe_cows(self, img, trainset):
        features = self.feature_extractor.image_to_embedding(img)
        distance = {}

        for key in self.feature_Map:
#             euclidean_distance = np.linalg.norm(self.feature_Map[key] - features)
#             distance[key] = euclidean_distance
            cos_distance = spatial.distance.cosine(self.feature_Map[key].flatten(), features.flatten())
            distance[key] = cos_distance
        closest_img_id = sorted(distance.items(), key=lambda kv: (str(kv[1]), str(kv[0])))[:2]

        results = {}
        for id, item in enumerate(closest_img_id):
            results[id] = [ item[0],str(item[1]), self.get_image(item[0], trainset)]

        return results
