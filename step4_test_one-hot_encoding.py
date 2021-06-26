import keras 
keras.__version__

import numpy as np


""" 
# 아래는 단어 수준의 윈-핫 인코딩

#초기 데이터. 하내의 샘플이 하나의 문
samples = ['The cat sat on the mat.', 'The dog ate my homework.'] # 이 때, The 는 중복되는 단어이지만, 대분자 The 와 소문자 the 는 다른 단어로 들어

# 데이터에 있는 모든 토큰 인덱스 구축
token_index = {}

for sample in samples:
    # split() : 샘플을 토큰으로 나눔
    # 여기선 그렇지 않으나, 실전에서 구두점과 특수 문자도 사용
    print("sample : ", sample)
    for word in sample.split():
        print("word : ", word)
        if word not in token_index:
            # 단어마다 고유 인텍스를 할당
            token_index[word] = len(token_index) + 1
            print("token_index" + "[" + word + "] :", token_index[word])
            
# 샘플을 벡터로 변환, 각 샘플에서 max_length 까지 단어만 사용 
max_length = 10

# result : 결과를 저장할 배열 
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
print(results) # 10x 11 matrix 가 2 개 만들어짐 
for i, sample in enumerate(samples):
    for j, word in list (enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i,j, index] = 1. # 각 word 는 11 차원 벡터로 표현됨 ex) 'The' = [0,1,0,0,0,0,0,0,0,0,0]
        
print(results)

"""

"""
#아래는 문자 수준의 원-핫 인코딩 
import string

samples = ['The cat sat on the mat.', 'The dog ate my homework.'] # 이 때, The 는 중복되는 단어이지만, 대분자 The 와 소문자 the 는 다른 단어로 들어
characters = string.printable #출력 가능한 모든 아스키 문자 
token_index = dict(zip(characters, range(1,len(characters)+1))) # 문자의 집합으로 인덱스 dictionary 만듦기

max_length = 50
results =np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, character in  enumerate(sample[:max_length]):
        index = token_index.get(character)
        print(index)
        results[i, j, index] = 1.
        
print(results)

"""

"""
# 아래는 케라스를 사용한 단어 수준의 원-핫 인코딩
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.'] 
tokenizer = Tokenizer(num_words=1000) # 가장 빈도가 높은 1000 개의 단어만 선택하도록 Tokenizer 객체를 만듦
tokenizer.fit_on_texts (samples) # 단어 인덱스를 구축

sequences = tokenizer.texts_to_sequences(samples) # 문자열을 정수 인텍스의 리스트로 변환
print(sequences)

one_hot_results = tokenizer.texts_to_matrix(samples, mode = 'binary') # 원-핫 이진 벡터 표현을 얻을 수 있음 

word_index = tokenizer.word_index 
print('Found %s unique tokens.' %len(word_index))

"""

#아래는 해싱 기법을 사용한 단어 수준의 원-핫 인코딩
samples = ['The cat sat on the mat.', 'The dog ate my homework.'] 

dimensionality = 1000 # 단어를 크기가 1000 인 벡터로 지정. 1000 개 (혹은 그 이상) 의 단어가 있다면 해싱 충돌이 늘어나고 인코딩 정확도가 감소
max_length = 10

results = np.zeros((len(samples), max_length, dimensionality )) 
for i, sample in enumerate(samples):
    print('i:',i)
    print('sample:', sample)
    for j ,word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality # 단어를 해싱하여 0 과 1000 사이의 랜덤한 정수 인텍스로 변환
        print(index)
        results[i,j,index]= 1.
