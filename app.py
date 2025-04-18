import os
from mtcnn import MTCNN
import streamlit as st
from PIL import Image
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from facenet_pytorch import InceptionResnetV1
import torch

filenames = pickle.load(open('features.pkl', 'rb'))
feature_list = pickle.load(open('embedding.pkl', 'rb'))
model = InceptionResnetV1(pretrained='vggface2').eval()
detector = MTCNN()


def save_ups(up_img):
    try:
        upload_path = os.path.join('uploads', up_img.name)
        with open(upload_path, 'wb') as f:
            f.write(up_img.getbuffer())
        return upload_path
    except:
        return None


def extract_features(img_path, model, detector):
    sample = cv2.imread(img_path)
    result = detector.detect_faces(sample)

    if not result:
        return None

    x, y, width, height = result[0]['box']
    face = sample[y:y + height, x:x + width]

    image = Image.fromarray(face)
    image = image.resize((160, 160))

    image = np.asarray(image).astype('float32')
    image = (image - 127.5) / 128.0

    img_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        result = model(img_tensor).cpu().numpy().flatten()
    return result


def recommend(feature_list, feature):
    sim = [cosine_similarity(feature.reshape(1, -1), f.reshape(1, -1))[0][0] for f in feature_list]
    idx = sorted(list(enumerate(sim)), reverse=True, key=lambda x: x[1])[0][0]
    return idx


st.title('Which Bollywood celebrity are you?')
up_img = st.file_uploader('Choose an Image')
if up_img is not None:
    upload_path = save_ups(up_img)
    if upload_path:
        disp = Image.open(upload_path)

        features = extract_features(upload_path, model, detector)
        if features is not None:
            idx = recommend(feature_list, features)
            actor = (" ".join(os.path.basename(filenames[idx]).split('_'))).split('.')[0]

            image_size = (300, 300)

            disp_resized = disp.resize(image_size)

            col1, col2 = st.columns(2)
            with col1:
                st.header('Uploaded image')
                st.image(disp_resized, width=image_size[0])
            with col2:
                st.header(actor)
                ref_image = Image.open(filenames[idx])
                ref_image_resized = ref_image.resize(image_size)
                st.image(ref_image_resized, width=image_size[0])
