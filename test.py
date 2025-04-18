from keras.applications.resnet50 import preprocess_input
import numpy as np
import pickle
from mtcnn import MTCNN
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from facenet_pytorch import InceptionResnetV1
import torch

feature_list = np.array(pickle.load(open('embedding.pkl', 'rb')))
filenames = pickle.load(open('features.pkl', 'rb'))

model = InceptionResnetV1(pretrained='vggface2').eval()

detector = MTCNN()

sample = cv2.imread('photos/Screenshot 2024-09-07 at 16.49.09.png')

result = detector.detect_faces(sample)

x, y, width, height = result[0]['box']
face = sample[y:y + height, x:x + width]

image = Image.fromarray(face)
image = image.resize((160, 160))

image = np.asarray(image).astype('float32')
image = (image - 127.5) / 128.0

img_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)

with torch.no_grad():
        result = model(img_tensor).cpu().numpy().flatten()

sim = [cosine_similarity(result.reshape(1, -1), feature.reshape(1, -1))[0][0] for feature in feature_list]

idx = sorted(list(enumerate(sim)), reverse=True, key=lambda x: x[1])[0][0]

temp_img = cv2.imread(filenames[idx])
cv2.imshow('Matched Image', temp_img)
cv2.waitKey(0)
