### Test

# 함수 준비하기

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import seaborn as sns

if type(tf.contrib) != type(tf):  # warning 출력 안하기
    tf.contrib._warning = None

# 한글 사용하기
import platform
from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus'] = False

if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~~')


# 데이터 불러오기
train = pd.read_csv('Data/2019-1st-ml-month-with-kakr/train.csv')
test = pd.read_csv('Data/2019-1st-ml-month-with-kakr/test.csv')

train.head()
test.head()

sns.set()
def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts() # feature에 따라 생존한 value(사람) 카운트
    dead = train[train['Survived'] == 0][feature].value_counts()     # feature에 따라 죽은 value(사람) 카운트
    df = pd.DataFrame([survived, dead])                              # 데이터프레임으로 묶기
    df.index = ['Survived', 'dead']                                  # 인덱스 달기
    df.plot(kind = 'bar', stacked = True, figsize = (10, 5))         # 차트 그리기


## 1. 성별에 따른 생존자 수
bar_chart('Sex')

# 여자들보다 남자들이 더 많이 죽었음


## 2. Pclass에 따른 생존자 수
bar_chart('Pclass')

# 3등석(값이 싼 등급)이 제일 많이 죽었음


## 3. 가족수에 따른 생존자 수
bar_chart('SibSp')

# 1명일 때(가족수가 혼자일때) 가장 많이 죽었음


## 4. 승선한 선착장에 따른 생존자 수
bar_chart('Embarked')

# S에서 탄 승객이 제일 많이 죽었음


data = [train, test] # train과 test 합치기
data

## 이름 전처리
train.head()

for dataset in data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)  # 정규식표현을 이용해 Mr. Mrs. 등 Title추출

train['Title'].value_counts()
test['Title'].value_counts()

title_mapping = {'Mr': 0, 'Miss': 1, 'Mrs': 2,
                 'Master': 3, 'Dr': 3, 'Rev': 3, 'Col': 3, 'Major': 3, 'Mlle': 3, 'Countess': 3,
                 'Ms': 3, 'Lady': 3, 'Jonkheer': 3, 'Don': 3, 'Dona': 3, 'Mme': 3, 'Capt': 3, 'Sir': 3}

for dataset in data:
    dataset['Title'] = dataset['Title'].map(title_mapping)

train.head()
train.info()
train.drop('Name', axis = 1, inplace = True)  # Name 컬럼을 지원준다
test.drop('Name', axis = 1, inplace = True)   # Name 컬럼을 지원준다
train.head()

bar_chart('Title')

## 성별 전처리
sex_mapping = {'male': 0, 'female': 1}        # 남자는 0, 여자는 1
for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
train.head()

bar_chart('Sex')


## 나이 빈칸을 나이 중앙값으로 처리
train['Age'].head()

# train의 Age 컬럼의 nan값을 train의 title로 group을 지어서 해당 그룹의 age컬럼의 median값으로 대체
train['Age'].fillna(train.groupby('Title')['Age'].transform('median'), inplace = True)
# test의 Age 컬럼의 nan값을 test의 title로 group을 지어서 해당 그룹의 age컬럼의 median값으로 대체
test['Age'].fillna(test.groupby('Title')['Age'].transform('median'), inplace = True)

train.head()

## 나이 전처리

for dataset in data:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0,                              # child = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,    # young = 0
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,    # adult = 2
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,    # mid-age = 3
    dataset.loc[dataset['Age'] > 62, 'Age'] = 4                                # senior = 4


train['Age'].head()
test['Age'].head()

bar_chart('Age')


## Embarked 전처리
Pclass1 = train[train['Pclass'] == 1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass'] == 2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass'] == 3]['Embarked'].value_counts()
# Embarked 컬럼에서 Pclass가 1일 인스턴스의 객수를 카운트해서 Pclass1변수에 담음
# 2,3 도 같이 반복

df= pd.DataFrame([Pclass1, Pclass2, Pclass3])             # Pclss 1, 2, 3 데이터 프레임 생성
df.index = ['1st class', '2nd class', '3rd class']        # 인덱스 만들어줌
df.plot(kind = 'bar', stacked = True, figsize = (10, 5))  # 그래프로 나타내기

# S에서 탑승한 승객이 많아서 NAN값은 S로 대체한다
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

train['Embarked']

embarked_mapping = {'S': 0 , 'C': 1, 'Q': 2}                        # 0, 1, 2 그룹으로 만들어줌
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
train.head()


## 운임 전처리
# NAN 값은 Median값으로 대체
train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'), inplace = True)
test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'), inplace = True)

train.head()

for dataset in data:
    dataset.loc[dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[dataset['Fare'] > 100, 'Fare'] = 3

train.head()


## Cabin 전처리
train.Cabin.value_counts()

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]

Pclass1 = train[train['Pclass'] == 1]['Cabin'].value_counts() # Pclass=1 에 해당하는 Cabin값은 카운트
Pclass2 = train[train['Pclass'] == 2]['Cabin'].value_counts() # Pclass=3 에 해당하는 Cabin값은 카운트
Pclass3 = train[train['Pclass'] == 3]['Cabin'].value_counts() # Pclass=2 에 해당하는 Cabin값은 카운트
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind = 'bar', stacked = True, figsize = (10, 5))

cabin_mapping = {'A': 0, 'B': 0.4, 'C': 0.8, 'D': 1.2, 'E': 1.6, 'F': 2, 'G': 2.4, 'T': 2.8}
for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

train['Cabin']

# Pclass 의 median으로 cabin 결측치 대체
train['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'), inplace = True)
test['Cabin'].fillna(test.groupby('Pclass')['Cabin'].transform('median'), inplace = True)
train['Cabin']


## 가족구성원수 전처리
# 형제자매수 + 부모자식수 + 나
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1 #sib = 형제자매, parch = 부모자식
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1 #sib = 형제자매, parch = 부모자식

train['FamilySize'].max()
test['FamilySize'].max()

# FamilySize의 범위는 1~11이라서 정규화를 시켜준다.
family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)

train.head()

# 티켓번호, 형제자매수, 부모가족수 컬럼은 삭제한다.
feature_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(feature_drop, axis = 1)
test = test.drop(feature_drop, axis = 1)
train = train.drop(['PassengerId'], axis = 1)
test = test.drop(['PassengerId'], axis = 1)

train.head()
train.info()
test.head()
test.info()

train = pd.DataFrame(train, columns = ['Pclass',
                                     'Sex',
                                     'Age',
                                     'Fare',
                                     'Cabin',
                                     'Embarked',
                                     'Title',
                                     'FamilySize'
                                     'Survived'])


train1 = train.values
test1 = test.values

X = train1[:, 1:]
Y = train1[:, 0]

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 모델 설정
model = Sequential()
model.add(Dense(32, input_dim = 8, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 모델 컴파일
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# 모델 저장 폴더 설정
MODEL_DIR = 'Model1/'
if not os.path.exists(MODEL_DIR):
   os.mkdir(MODEL_DIR)

# 모델 저장 조건 설정
modelpath = "Model1/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath = modelpath,
                               monitor = 'val_loss',
                               verbose = 1,
                               save_best_only = True)

# 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor = 'val_loss',
                                        patience = 100)

# 모델 실행 및 저장
history = model.fit(X, Y,
                    validation_split = 0.2,
                    epochs = 500,
                    batch_size = 5,
                    verbose = 0,
                    callbacks = [early_stopping_callback,checkpointer])

# 결과를 출력합니다.
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))

# 테스트 셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']

# 그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker = '.', c = "red", label = 'Testset_loss')
plt.plot(x_len, y_loss, marker = '.', c = "blue", label = 'Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# 모델 프레임 설정
# 은닉층 노드 32, 속성 8, 함수 relu
# 은닉층 노드 16, 함수 relu
# 은닉층 노드 8, 함수 relu
# 출력층 노드 1, 함수 softmax

# 이 모델의 정확도는 87.54이고
# 에포크 500번 실행시 173번째가 가장 좋은 정확도를 가졌다.
