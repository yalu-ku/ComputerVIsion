import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

import cv2
import os
import shutil


def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


## 데이터 분류
path_prefix = 'D:/GITHUB/ComputerVision/AlexNet/flowers/'
model_prefix = 'D:/GITHUB/ComputerVision/AlexNet/210513/'

labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

train_path = model_prefix + 'train'
valid_path = model_prefix + 'valid'
test_path = model_prefix + 'test'

make_dir(model_prefix)
make_dir(train_path)
make_dir(valid_path)
make_dir(test_path)

folder_list = [train_path, valid_path, test_path]

for folder_path in folder_list:
    for label in labels:
        temp_path = folder_path + '/' + label
        make_dir(temp_path)

for label in labels:
    image_list = os.listdir(path_prefix + '/' + label)
    image_num = len(image_list)
    print(label + ' folder has ' + str(image_num) + ' images.')

    ## 5 : 3 : 2
    train_index = int(image_num * 0.5)
    test_index = int(image_num * 0.8)

    for image in image_list[:train_index]:
        shutil.copy(path_prefix + label + '/' + image, train_path + '/' + label + '/' + image)
    for image in image_list[train_index:test_index]:
        shutil.copy(path_prefix + label + '/' + image, valid_path + '/' + label + '/' + image)
    for image in image_list[test_index:]:
        shutil.copy(path_prefix + label + '/' + image, test_path + '/' + label + '/' + image)

    print("image moved")

## Data Augmentation (just trainSet)
train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.2, height_shift_range=0.2,
                                   zoom_range=0.2, horizontal_flip=True, vertical_flip=True)
train_generator = train_datagen.flow_from_directory(
    train_path, target_size=(227,227),
    batch_size=128,
    class_mode='categorical'
)

valid_datagen = ImageDataGenerator(rescale=1./255)
valid_geneator = valid_datagen.flow_from_directory(
    valid_path,
    target_size=(277,277),
    batch_size=128,
    class_mode='categorical')

## testSet은 절때 이미지 가공하지 않는다.
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(277,277),
    batch_size=128,
    class_mode='categorical')


## 데이터 전처리 완료
#########################################################


## AlexNet 구조 직접 만들어보기
def AlexNet():
    # 간단한 모델이기 때문에
    model = Sequential()
    model.input_shape(277,277,3)
    # Conv2D(필터의 개수, (커널 사이즈 : 연산량 줄이기 위해),
    model.add(Conv2D(96, (11,11), strides=4,
              padding='valid', activation='relu'))  # 사이즈가 줄어드는 valid
    model.add(MaxPooling2D(pool_size=(3,3), strides=2))
    model.add(BatchNormalization())
    # 하나의 싸이클

    model.add(Conv2D(256, (5,5), strides=1, padding='same', activation='relu'))     # 패딩이 same이면 input==output 사이즈 동일
    model.add(MaxPooling2D(pool_size=(3,3), stride=2))  # 위와 동일하게 반복
    model.add(BatchNormalization())

    model.add(Conv2D(384, (3,3), strides=1, padding='same', activation='relu'))
    model.add(Conv2D(384, (3,3), strides=1, padding='same', activation='relu'))
    model.add(Conv2D(384, (3,3), strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=2))
    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    return model

model = AlexNet()
opt = SGD(lr=0.01, decay=5e-4, momentum=0.9)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()


