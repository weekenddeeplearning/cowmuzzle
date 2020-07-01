import os
import cv2
import numpy as np
import pickle
import base64
from scipy import spatial

class cowFinder():
    def loadFeature(self, path):
        pickle_in = open(path, "rb")
        feature_Map = pickle.load(pickle_in)
        return feature_Map

    def load_extractor(self):
        tensorflowNet = cv2.dnn.readNetFromTensorflow('./frozen_models/frozen_graph.pb')
        return tensorflowNet

    def get_image_blob(self,img):
        blob = cv2.dnn.blobFromImage(img, 1, (224, 224), swapRB=True)
        return blob

    def __init__(self):
        self.feature_Path = "feature_map_dnn.pickle"
        self.feature_Map = self.loadFeature(self.feature_Path)
        self.feature_extractor = self.load_extractor()

    def get_image(self, closet, trainset):
        with open(os.path.join(trainset,closet), "rb") as img_file:
            text_bytes = base64.b64encode(img_file.read())
            base64_string = text_bytes.decode('utf-8')
        return base64_string

    def find_top_two_cows(self, img, trainset):
        blob = self.get_image_blob(img)
        self.feature_extractor.setInput(cv2.dnn.blobFromImage(img, size=(224, 224), swapRB=False, crop=False))
        features = self.feature_extractor.forward()
        distance = {}

        for key in self.feature_Map:
            # euclidean_distance = np.linalg.norm(self.feature_Map[key] - features)
            # distance[key] = euclidean_distance
            cos_distance = spatial.distance.cosine(self.feature_Map[key].flatten(), features.flatten())
            distance[key] = cos_distance

        closest_img_id = sorted(distance.items(), key=lambda kv: (str(kv[1]), str(kv[0])))[:2]

        results = {}
        for id, item in enumerate(closest_img_id):
            results[id] = [ item[0],str(item[1]), self.get_image(item[0], trainset)]

        return results
