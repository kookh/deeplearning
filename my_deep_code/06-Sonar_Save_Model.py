# 초음파 광물 예측하기 : 모델 저장과 재사용

from keras.models import Sequential, load_model
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder


import pandas as pd
import numpy
import tensorflow as tf


seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 입력
df = pd.read_csv('../dataset/sonar.csv', header=None)

dataset = df.values
X = dataset[:, 0:60]
Y_obj = dataset[:,60]


# 문자열을 숫자로 변환
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# 학습셋과 테스트셋의 구분
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

# 모델의 설정
model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss = 'mean_squared_error', optimizer='adam', metrics=['accuracy'])

# 모델 실행
model.fit(X_train, Y_train, epochs=130, batch_size=5)
model.save('my_model.h5')

del model
model =  load_model('my_model.h5')

print("\n Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))