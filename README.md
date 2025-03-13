# 약물 남용 방지 정보 제공 시스템

- 2024.09.01 ~ 2024.05.20
- NEO-FFI-R 이라는 정밀한 성격 유형 검사를 통해 자신의 약물 소비 위험도 및 관련 약물을 알아보고 자신이 소지한 약물이 위험 약물인지 판별해주는 시스템을 개발함

> **핵심전략**
> 
> 1. **데이터 불균형 문제 해결**
>     - 문제 : 다중분류 모델에서 데이터 불균형으로 성능 저조
>     1. SMOTE, 클래스 가중치 조정, 언더샘플링, 앙상블 학습을 적용하여 불균형 문제를 해결하고 성능 향상
> 2. **위험도 점수 생성**
>     - 문제 : 객관적인 위험도 점수 척도가 없어 설문 결과로 점수를 매기기 어려움
>     1. NEO-FFI-R 성격 검사 결과를 바탕으로 가중치를 설정하고 다차원 척도법(MDS, Multidimensional Scaling)으로 위험도 점수를 생성

### 1. 개발 배경

- 최근 마약 사범이 급증하고 있으며, 10~20대가 전체의 1/3을 차지한다는 통계가 보도됨
- 청소년기는 판단력이 부족해 호기심과 사회적 압박에 쉽게 영향을 받아 위험한 선택을 할 가능성이 높음
    - 따라서 이들에게 정확한 정보와 경각심을 심어주는 교육이 필요함
- 타인의 고의에 의해서 혹은 본인의 실수로 인해서 해로운 약을 식별하지 못할 수 있음
    - 따라서 약물을 자동으로 식별하여 해당 약물이 무엇인지와 그에 대한 정보를 알려주는 기술이 필요함

### 2. 기대 효과

- 청소년 마약 범죄 예방
    - 10대~20대에 유행하는 성격 유형 검사로 이목을 집중시켜 마약의 위험성을 제시하여 청소년 마약 범죄 예방에 도움을 줌
    - 또한, 개인 성격에 따른 마약 위험도를 분석하고 성격 요인별로 접하기 쉬운 마약 성분을 예측하여 정보를 제공
- 약물 강제 투약 피해 예방
    - 컴퓨터 비전을 통한 알약 자동 식별로 마약류 여부를 구별하여 추가 마약 범죄 피해를 예방

### 3. 아키텍쳐

- AI 분석 모듈 구성 및 예측 모델 데이터 저장(DB) → 사용자 서비스 DB 구현 → 로그인 API 및 계정 DB 구현 → 사용자 서비스 리스너 구축 → 웹 페이지 구현 (사용자 서비스와 연결) → 관리자 서비스2 리스너 구축 → 모델 관리 시스템 구축
- 데이터 수집 및 전처리 → 모델링(MLPC, RFC, GBC, SVMC, LDAC, NBC, Logistic Regression) → 모델 학습 및 평가(Accuracy, recall, precision, f1 score) → 하이퍼 파라미터 튜닝(GA)을 통한 최적화

### 4. 사용 기술

- Python, Data Preprocessing, TensorFlow, PyTorch, Fast API, Uvicorn

### 5. 라이브러리

- Numpy, Scipy, Pandas, Scikit-learn, Matplotlib, Seaborn, TensorFlow, PyTorch, FastAPI, Uvicorn

### 6. 개발 환경

- 언어 : Python 3.10.9
- OS : Window 11
- IDE : Jupyter Notebook, VS Code

<aside>


### **🚀성과**

- **Neo-FFI-R 성격 유형 검사를 활용한 머신러닝 기반 마약 소비 위험도 예측 및 종류 분류** ([학술지 게재](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11652063))
- **대학생 논문 경진대회 금상**
- **연구 성과 경진 대회 및 제 33회 소프트웨어 전시회 대상**
</aside>
