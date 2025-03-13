import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# GPU 사용
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

def preprocess_image(image_path):
    # 이미지 로딩
    image = cv2.imread(image_path)
    # 이미지 크기 조정
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # CLAHE를 적용하여 이미지 평탄화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 차원 추가 (CNN 모델에 입력하기 위해)
    image = np.expand_dims(image, axis=0)
    return image

image_directory = '/home/jongin/user/proj_pill/pill_data/pill_data_croped'
classes = [folder for folder in os.listdir(image_directory) if 'json' not in folder]

# 이미지 데이터와 라벨을 저장할 리스트 초기화
data = []
labels = []

# 이미지 데이터 로드 및 전처리
for idx, class_name in enumerate(classes):
    class_path = os.path.join(image_directory, class_name)
    print(class_path)
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        # image = cv2.imread(image_path)
        # image = np.expand_dims(image, axis=0)
        image = preprocess_image(image_path)  # 이미지 전처리 추가
        data.append(image)
        labels.append(idx)

# 데이터셋을 NumPy 배열로 변환
data = np.concatenate(data, axis=0)
labels = np.array(labels)

# 라벨을 원-핫 인코딩
labels = to_categorical(labels)

# 훈련 데이터와 테스트 데이터로 분할
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# CNN 모델 생성
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련을 위한 OneDeviceStrategy 설정
strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')

# 모델 생성과 컴파일을 OneDeviceStrategy 내에서 수행
with strategy.scope():
    # CNN 모델 생성
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(classes), activation='softmax'))

    # 모델 컴파일
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 학습 파라미터 설정
batch_size = 32  # 한 번에 처리할 배치 크기
num_epochs = 10  # 전체 데이터셋에 대해 반복할 에포크 수

# 데이터셋을 배치로 나누어 학습
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    # 배치 데이터 생성 및 학습
    for batch_start in range(0, len(train_data), batch_size):
        batch_end = batch_start + batch_size
        batch_data = train_data[batch_start:batch_end]
        batch_labels = train_labels[batch_start:batch_end]
        
        # 모델 훈련
        model.train_on_batch(batch_data, batch_labels)
        
    # 에포크별 평가
    loss, accuracy = model.evaluate(test_data, test_labels)
    print(f"Validation Accuracy: {accuracy*100:.2f}%")

# 모델 저장
model.save('model.h5')
