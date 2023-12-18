from typing import Dict, Union
import warnings
from numpy import expand_dims
from sklearn.preprocessing import Normalizer
import keras
from .inception_resnet import InceptionResNetV1
import numpy as np 
import gdown
import cv2 
import kserve 
import logging 
from kserve.utils.utils import get_predict_input, get_predict_response
from kserve.errors import InferenceError, ModelMissingError
from kserve import ModelServer, model_server, InferRequest, InferOutput, InferResponse

warnings.filterwarnings('ignore')

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)
def normalize(data):
    in_encoder = Normalizer(norm='l2')
    data = in_encoder.transform(data)
    print()
    return data

class EmbeddingModel(kserve.Model): 
    def __init__(self, name: str, model_dir: str) -> None:
        super().__init__(name)
        self.name = name
        self.model_dir = model_dir
        self.load() 

    def download_weights(self):
        print("downloading weights....")
        try:
            url = "https://drive.google.com/drive/folders/1q8vnxtso2RLuzjr1mdMFMinKKcPv3jSX?usp=sharing"
            gdown.download_folder(url, quiet=True, use_cookies=False)
            print("weights downloaded successfully")
        except Exception as e:
            raise RuntimeError("some error occured while downloading weights.", e)

    def load(self):
        self.download_weights()
        self.vanilla_embedding_model = InceptionResNetV1(weights_path='verifier_weights/facenet_keras_weights_VGGFace2.h5', classes=512)
        self.embedding_model = keras.models.Model(self.vanilla_embedding_model.inputs, self.vanilla_embedding_model.layers[-1].output)
        self.ready = True 

    def preprocess(self, payload, headers):
        image_list = get_predict_input(payload)
        # image_list = request.get_json()["image_array"]
    
        arrImg = np.array(image_list)

        frame = cv2.cvtColor(np.float32(arrImg), cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame,(160,160),cv2.INTER_AREA)
        face_frame = np.array(frame)
        # embedding = get_embedding(frame)

        face_frame = face_frame.astype('float32')
        mean, std = face_frame.mean(), face_frame.std()
        face_frame = (face_frame - mean) / std
        face_frame = expand_dims(face_frame, axis=0)
        print()
        return face_frame

    def predict(self, payload, headers):
        try:
            embedding = self.embedding_model.predict(payload)
            return embedding 
        except Exception as e:
            raise InferenceError(str(e))

    def postprocess(self, payload, headers)-> Union[Dict, InferResponse]:
        result = normalize([payload[0]])
        infer_output = InferOutput(name="output-0", shape=list(result.shape), datatype="FP32", data=result)
        return InferResponse(model_name=self.name, infer_outputs=[infer_output], response_id=123)

