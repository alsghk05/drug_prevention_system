from transformers import BertTokenizerFast, GPT2LMHeadModel

# 모델 초기화 및 로드
tokenizer_ko = BertTokenizerFast.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")
model_ko = GPT2LMHeadModel.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")
model_ko.eval()  # 모델을 평가 모드로 설정

# 사용자 대화 이력 저장소
user_history = {}

# 사용자 성격 유형 검사 결과 저장소
user_results = {}

def update_user_results(user_id, risk_level, drug_types):
    # 사용자 성격 유형 검사 결과 업데이트
    user_results[user_id] = {
        "risk_level": risk_level,
        "drug_types": drug_types
    }

def get_user_info(user_id):
    # 사용자 성격 유형 검사 결과 반환
    result = user_results.get(user_id, None)
    if result:
        return f"당신의 마약 소비 위험도는 {result['risk_level']}%입니다. 주로 소비하는 마약 종류는 {', '.join(result['drug_types'])}입니다."
    else:
        return "아직 성격 유형 검사 결과가 입력되지 않았습니다."

def chatbot(user_id, input_text):
    # 사용자 대화 이력을 바탕으로 현재 대화 문맥 생성
    context = " ".join(user_history.get(user_id, [])) + input_text
    
    # 입력 텍스트를 토크나이저로 인코딩
    encoded_input = tokenizer_ko(context, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # 모델을 사용하여 응답 생성
    output_ids = model_ko.generate(
        encoded_input['input_ids'],
        attention_mask=encoded_input['attention_mask'],
        max_length=100,  # 최대 출력 길이 설정
        num_return_sequences=1,  # 반환할 응답 수 설정
        no_repeat_ngram_size=2,  # 중복 방지 설정
        pad_token_id=tokenizer_ko.eos_token_id  # 패딩 토큰 ID 설정
    )
    
    # 생성된 응답을 디코딩하여 텍스트로 변환
    chat_output = tokenizer_ko.decode(output_ids[0], skip_special_tokens=True)
    
    # 사용자 대화 이력에 새로운 응답 추가
    user_history[user_id] = user_history.get(user_id, []) + [chat_output]
    user_history[user_id] = user_history[user_id][-5:]  # 최근 5개의 대화만 저장하여 이력 관리
    
    return chat_output  # 챗봇 응답 반환

if __name__ == "__main__":
    while True:
        user_input = input("사용자: ")  # 사용자 입력 받기
        if user_input.lower() == "quit":
            break  # 'quit' 입력 시 종료
        elif user_input.lower().startswith("결과: "):
            # 성격 유형 검사 결과 업데이트
            result_data = user_input[len("결과: "):]
            parts = result_data.split(';')
            if len(parts) == 2:
                risk_level = parts[0].strip()
                drug_types = parts[1].strip().split(',')
                update_user_results("user123", risk_level, drug_types)
                print("성격 유형 검사 결과가 업데이트되었습니다.")
            else:
                print("결과 입력 형식이 잘못되었습니다. 예시: '결과: 18.26; 코카인, 대마초, 엑스터시'")
        elif user_input.lower() == "정보":
            # 성격 유형 검사 결과 정보 제공
            user_info = get_user_info("user123")
            print("챗봇: " + user_info)
        else:
            response = chatbot("user123", user_input)  # 챗봇 응답 생성
            print("챗봇: " + response)  # 챗봇 응답 출력
