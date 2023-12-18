# from .model import EmbeddingModel
# import numpy as np 
# import cv2 
# from flask import Flask, request
# from PIL import Image
# import json

# embedding_model = EmbeddingModel()

# def get_embedding(face_frame):
    
#     embedding = embedding_model.load_face(face_frame).preprocess().predict().postprocess() 
#     return embedding

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return {"text": "hello verifier"}

# @app.route('/predict', methods=['POST'])
# def upload():

#     image_list = request.get_json()["image_array"]
#     arrImg = np.array(image_list)

#     frame = cv2.cvtColor(np.float32(arrImg), cv2.COLOR_RGB2BGR)
#     frame = cv2.resize(frame,(160,160),cv2.INTER_AREA)
#     frame = np.array(frame)
#     embedding = get_embedding(frame)
    
#     return {"embedding": embedding.tolist()}

# if __name__ == '__main__':
#     app.run(debug=True, port=4001, host="0.0.0.0")

# # if __name__ == "__main__":
# #     x0, y0, x1, y1 = map(int, [p1[0], p1[1], p2[0], p2[1]])
# #     im = np.array(frame[x0:x1,y0:y1,:])
# #     im = cv2.resize(im,(160,160),cv2.INTER_AREA)
# #     item = np.array(im)
# #     print(get_embedding(item))