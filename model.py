from numpy import expand_dims
from sklearn.preprocessing import Normalizer
from keras.models import Model
from inception_resnet import InceptionResNetV1

def normalize(data):
    in_encoder = Normalizer(norm='l2')
    data = in_encoder.transform(data)
    print()
    return data

class EmbeddingModel: 
    def __init__(self) -> None:
        self.load() 
        self.face_frame = None 
    def load(self):
        self.vanilla_embedding_model = InceptionResNetV1(weights_path='weights/facenet_keras_weights_VGGFace2.h5', classes=512)
        self.embedding_model = Model(self.vanilla_embedding_model.inputs, self.vanilla_embedding_model.layers[-1].output)
    
    def load_face(self, face_frame):
        self.face_frame = face_frame 
        return self 

    def preprocess(self):
        self.face_frame = self.face_frame.astype('float32')
        mean, std = self.face_frame.mean(), self.face_frame.std()
        self.face_frame = (self.face_frame - mean) / std
        self.face_frame = expand_dims(self.face_frame, axis=0)
        return self

    def predict(self):
        self.embedding = self.embedding_model.predict(self.face_frame)
        return self 

    def postprocess(self):
        return normalize([self.embedding[0]])

