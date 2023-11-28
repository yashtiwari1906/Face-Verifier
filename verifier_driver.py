from model import EmbeddingModel
import numpy as np 
import cv2 
from flask import Flask, request
from PIL import Image
import json

def get_embedding(face_frame):
    embedding_model = EmbeddingModel()
    embedding = embedding_model.load_face(face_frame).preprocess().predict().postprocess() 
    return embedding

app = Flask(__name__)

@app.route('/')
def home():
    return {"text": "hello verifier"}

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    img = Image.open(file).convert('RGB')
    arrImg = np.array(img)

    frame = cv2.cvtColor(arrImg, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame,(160,160),cv2.INTER_AREA)
    frame = np.array(frame)
    embedding = get_embedding(frame)
    if file.filename == '':
        return 'No selected file'

    if file:
        return {"mode": img.mode, "size": img.size, "embedding": embedding.tolist()}
if __name__ == '__main__':
    app.run(debug=True, port=4001)

# if __name__ == "__main__":
#     x0, y0, x1, y1 = map(int, [p1[0], p1[1], p2[0], p2[1]])
#     im = np.array(frame[x0:x1,y0:y1,:])
#     im = cv2.resize(im,(160,160),cv2.INTER_AREA)
#     item = np.array(im)
#     print(get_embedding(item))