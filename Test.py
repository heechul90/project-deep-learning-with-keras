### Test

# 함수 준비하기

from keras.models import Sequential  # 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옵니다.
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

# 모듈 준비하기

import numpy as np                # 필요한 라이브러리를 불러옵니다.
import pandas as pd
import re
import os
import tensorflow as tf
import matplotlib.pyplot as plt
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

test = pd.read_csv('Data/2019-1st-ml-month-with-kakr/test.csv',
                    header = 0,
                    dtype = {'Age' : np.float64})

train = pd.read_csv('Data/2019-1st-ml-month-with-kakr/train.csv',
                    header = 0,
                    dtype = {'Age' : np.float64})

train.head()
train.columns
train.info()

titanic = [train, test]


## pclass별 생존자
pclass_survived = train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).sum()
pclass_survived

plt.bar(pclass_survived['Pclass'], pclass_survived['Survived'])
plt.title('Pclass별 생존자')
plt.xlabel('Pclass')
plt.ylabel('Survived')
plt.show()

# Pclass별 생존자 수는 1, 3, 2 순으로 생존자수가 많았다.

## 성별 생존자
sex_survived = train[['Sex', 'Survived']].groupby(['Sex'], as_index = False).sum()
sex_survived

plt.bar(sex_survived['Sex'], sex_survived['Survived'])
plt.title('Sex별 생존자')
plt.xlabel('Sex')
plt.ylabel('Survived')
plt.show()

# 성별 생존자는 여성이 남성보다 많았다.


## familysize별 생존자
for dataset in titanic:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
familysize_survived = train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).sum()

plt.bar(familysize_survived['FamilySize'], familysize_survived['Survived'])
plt.title('familysize별 생존자')
plt.xlabel('familysize')
plt.ylabel('Survived')
plt.show()

# Familysize별 생존자는 구성원이 1일때 가장 많이 살았다.
for dataset in titanic:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).sum())


## Embarked별 생존자
for dataset in titanic:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
embark_survived = train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).sum()

plt.bar(embark_survived['Embarked'], embark_survived['Survived'])
plt.title('Embarked별 생존자')
plt.xlabel('Embarked')
plt.ylabel('Survived')
plt.show()

# Embark별 생존자는 S가 가장 많음

## Fare별 생존자
for dataset in titanic:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
fare_survived = train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean()

plt.bar(fare_survived['CategoricalFare'], fare_survived['Survived'])
plt.title('CategoricalFare별 생존자')
plt.xlabel('CategoricalFare')
plt.ylabel('Survived')
plt.show()

## 나이별 생존자
for dataset in titanic:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)

print(train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())


## 이름
def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

for dataset in titanic:
    dataset['Title'] = dataset['Name'].apply(get_title)

print(pd.crosstab(train['Title'], train['Sex']))

for dataset in titanic:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

name_survived = train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

plt.bar(name_survived['Title'], name_survived['Survived'])
plt.title('Title별 생존자')
plt.xlabel('Title')
plt.ylabel('Survived')
plt.show()


## 전처리
for dataset in titanic:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # Mapping Fare
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    # Mapping Age
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

# Feature Selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', \
                 'Parch', 'FamilySize']
train = train.drop(drop_elements, axis=1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis=1)

test = test.drop(drop_elements, axis=1)

print(train.head(10))

train = train.values
test = test.values

X = train[:, :11]
Y = train[:, 11]

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 모델 설정
model = Sequential()
model.add(Dense(32, input_dim = 11, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 모델 컴파일
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# 모델 실행
model.fit(X, Y,
          epochs = 200,
          batch_size = 200)



data = pd.DataFrame(data, columns = ['PassengerId',
                                     'Pclass',
                                     'Name',
                                     'Sex',
                                     'Age',
                                     'SibSp',
                                     'Parch',
                                     'Ticket',
                                     'Fare',
                                     'Cabin',
                                     'Embarked',
                                     'Survived'])