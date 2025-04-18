from keras.preprocessing import image
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
import os
import pickle

base_directory = 'Bollywood_celeb_face_localized'

files = [
    os.path.join(base_directory, folder, actor, file)
    for folder in os.listdir(base_directory)
    for actor in os.listdir(os.path.join(base_directory, folder))
    for file in os.listdir(os.path.join(base_directory, folder, actor))
]

pickle.dump(files, open('features.pkl', 'wb'))

filenames = pickle.load(open('features.pkl', 'rb'))

model = InceptionResnetV1(pretrained='vggface2').eval()

def feature_extractor(img_path, model):
    img = image.load_img(img_path, target_size=(160, 160))
    img_arr = image.img_to_array(img)

    img_tensor = torch.tensor(img_arr).permute(2, 0, 1).unsqueeze(0)
    img_tensor = (img_tensor - 127.5) / 128.0

    with torch.no_grad():
        result = model(img_tensor).cpu().numpy().flatten()

    return result

features = []
for file in tqdm(filenames):
    features.append(feature_extractor(file, model))

pickle.dump(features, open('embedding.pkl', 'wb'))
