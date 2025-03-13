# train_model.py
from transformers import TrainingArguments, Trainer
from torch.utils.data import Dataset
from chatbot import tokenizer_ko, model_ko  # chatbot.py에서 토크나이저와 모델 가져오기
import torch

# 사용자 정의 데이터셋 클래스 정의
class MyDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        # 입력 텍스트를 토크나이저로 인코딩
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)

    def __getitem__(self, idx):
        # 인덱스에 해당하는 샘플 반환
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        # 데이터셋의 길이 반환
        return len(self.encodings.input_ids)

# 데이터 로드 함수 정의
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()  # 파일의 모든 줄을 읽어옴
    return lines

# 학습할 텍스트 파일 로드
texts = load_data('./NEO-FFI-R.txt')  # 텍스트 파일에서 데이터 읽어오기
dataset = MyDataset(texts, tokenizer_ko)  # 데이터셋 객체 생성

# 훈련 설정 정의
training_args = TrainingArguments(
    output_dir='./results',  # 모델 출력 디렉토리
    num_train_epochs=3,  # 학습 에포크 수
    per_device_train_batch_size=4,  # 각 장치당 배치 크기
    warmup_steps=500,  # 워밍업 스텝 수
    weight_decay=0.01,  # 가중치 감쇠
    logging_dir='./logs',  # 로깅 디렉토리
    logging_steps=10,  # 로깅 스텝 간격
)

# Trainer 객체 생성
trainer = Trainer(
    model=model_ko,  # 학습할 모델
    args=training_args,  # 훈련 설정
    train_dataset=dataset  # 학습 데이터셋
)

# 메인 함수 정의
if __name__ == "__main__":
    trainer.train()  # 모델 학습 시작
