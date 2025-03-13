#!/usr/bin/env python
# coding: utf-8

# ## 데이터 불러오기

# In[1]:


import pandas as pd

df = pd.read_csv('drug_consumption_replace.csv')
df.head()


# In[2]:


df = df.replace('CL0', 0).replace('CL1', 1).replace('CL2', 2).replace('CL3', 3).replace('CL4', 4).replace('CL5', 5).replace('CL6', 6)
df


# In[3]:


df.columns


# In[4]:


df = df.drop(['Unnamed: 0', 'ID', 'Age', 'Gender', 'Education', 'Ethnicity', 'Impulsive', 'SS'], axis=1)
df.head()


# ## 데이터 전처리
# - df_features에 성격 검사 점수를 넣고, df_labels에 각 마약 종류의 소비 빈도를 넣음

# In[5]:


df_features = df[['Escore', 'Oscore', 'Ascore', 'Cscore', 'Nscore']]
df_labels = df[['Amphet', 'Amyl', 'Benzos', 'Cannabis', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine',
       'Legalh', 'LSD', 'Meth', 'Nicotine', 'Semer', 'VSA']]


# ## 데이터 정규화

# In[6]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Perform standardization on the  features
df_features_scaled = scaler.fit_transform(df_features)

# Convert scaled  features back to DataFrame for better readability
df_features_scaled = pd.DataFrame(df_features_scaled, columns=df_features.columns)

# Display first few rows of scaled features
df_features_scaled.head()


# ### 학습 데이터, 테스트 데이터 분할

# In[7]:


from sklearn.model_selection import train_test_split
import random
import numpy as np

# 랜덤 시드 설정
np.random.seed(42)
random.seed(42)

# 데이터를 학습 데이터와 테스트 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(df_features_scaled, df_labels, test_size=0.2, random_state=42)

# 이후에도 같은 분할을 유지하려면 random_state를 고정합니다 (여기서는 42로 설정)


# In[8]:


y_test


# ## 1. 모델 학습
# ### 1) Neural Net 모델 예측

# In[9]:


from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.neural_network import MLPClassifier

## 지금 최종최종최종
mlp_clf_tuned = MLPClassifier(
    activation = 'relu',
    solver = 'adam',
    hidden_layer_sizes = (50,),
    alpha = 0.06357208758199738,
    random_state=42
)

# MultiOutputClassifier로 MLPClassifier 래핑
multi_model = MultiOutputClassifier(mlp_clf_tuned)

# 모델 학습
multi_model.fit(X_train, y_train)

# 예측 수행
y_pred_multi_mlp = multi_model.predict(X_test)

# 결과 저장을 위한 딕셔너리 초기화
multioutput_results_mlp_tuned = {}

# Numpy 배열로 변환
y_test_np = y_test.values
y_pred_multi_mlp_np = y_pred_multi_mlp

# 각 레이블에 대한 지표 계산 후 평균 취하기
accuracy_avg_mlp_tuned = np.mean([accuracy_score(y_test_np[:, i], y_pred_multi_mlp_np[:, i])
                                  for i in range(y_test_np.shape[1])])
f1_avg_mlp_tuned = np.mean([f1_score(y_test_np[:, i], y_pred_multi_mlp_np[:, i], average='weighted')
                            for i in range(y_test_np.shape[1])])
recall_avg_mlp_tuned = np.mean([recall_score(y_test_np[:, i], y_pred_multi_mlp_np[:, i], average='weighted')
                                for i in range(y_test_np.shape[1])])
precision_avg_mlp_tuned = np.mean([precision_score(y_test_np[:, i], y_pred_multi_mlp_np[:, i], average='weighted')
                                   for i in range(y_test_np.shape[1])])

# 결과 저장
multioutput_results_mlp_tuned['MLPClassifier'] = {
    'Accuracy': accuracy_avg_mlp_tuned,
    'F1 Score': f1_avg_mlp_tuned,
    'Recall': recall_avg_mlp_tuned,
    'Precision': precision_avg_mlp_tuned
}

print("Tuned MLPClassifier Multi-output Classification Results:")
print(multioutput_results_mlp_tuned)


# In[12]:


pred_result = pd.DataFrame(y_pred_multi_mlp_np, columns = ['Amphet', 'Amyl', 'Benzos', 'Cannabis', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine',
       'Legalh', 'LSD', 'Meth', 'Nicotine', 'Semer', 'VSA'])
pred_result


# ### 점수 높은 상위 3개 마약 출력

# In[14]:


selected_columns_per_row = []

for index, row in pred_result.iterrows():
    selected_columns = row.index[row.ge(5)].tolist()
    selected_columns_per_row.append(selected_columns)

selected_columns_per_row = pd.DataFrame(selected_columns_per_row)
selected_columns_per_row


# In[ ]:




