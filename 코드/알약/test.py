import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import json

# 저장된 모델 로드
loaded_model = load_model('model.h5')

root_dir = "/home/jongin/user/proj_pill/pill_data/pill_data_croped/"
image_directory = '/home/jongin/user/proj_pill/pill_data/pill_data_croped'
classes = [folder for folder in os.listdir(image_directory) if 'json' not in folder]

# 이미지 파일 경로
image_path = '/home/jongin/user/proj_pill/pill_data/pill_data_croped/K-000364/K-000364_0_0_0_0_60_040_200.png'

def preprocess_image(image_path):
    # 이미지 로드
    image = cv2.imread(image_path)
    # 이미지 크기 조정
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # CLAHE를 적용하여 이미지 평탄화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 차원 추가 (CNN 모델에 입력하기 위해)
    image = np.expand_dims(image, axis=-1)
    return image

# 이미지 전처리
input_image = preprocess_image(image_path)

# 예측
predictions = loaded_model.predict(np.array([input_image]))
# 가장 높은 확률을 가진 클래스 인덱스 찾기
predicted_class_index = np.argmax(predictions[0])
# 예측 결과 출력
print('Predicted class index:', classes[predicted_class_index])

img_list = os.listdir(os.path.join(root_dir, classes[predicted_class_index]))
json_path = os.path.join(root_dir, classes[predicted_class_index]+'_json', img_list[0].replace('.png', '.json'))
with open(json_path, 'r') as f:
  json_data = json.load(f)
  pill_name = json_data['images'][0]['dl_name']
print(pill_name)
