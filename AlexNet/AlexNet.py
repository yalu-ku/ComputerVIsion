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
    target_size=(227,227),
    batch_size=128,
    class_mode='categorical')

## testSet은 절때 이미지 가공하지 않는다.
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(227,227),
    batch_size=128,
    class_mode='categorical')


## 데이터 전처리 완료
#########################################################


## AlexNet 구조 직접 만들어보기
def AlexNet():
    # 간단한 모델이기 때문에
    model = Sequential()

    # Conv2D(필터의 개수, (커널 사이즈 : 연산량 줄이기 위해),
    model.add(Conv2D(96, (11,11), strides=4, input_shape=(227,227,3),
              padding='valid', activation='relu'))  # 사이즈가 줄어드는 valid
    model.add(MaxPooling2D(pool_size=(3,3), strides=2))
    model.add(BatchNormalization())
    # 하나의 싸이클

    model.add(Conv2D(256, (5,5), strides=1, padding='same', activation='relu'))     # 패딩이 same이면 input==output 사이즈 동일
    model.add(MaxPooling2D(pool_size=(3,3), strides=2))  # 위와 동일하게 반복
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


## Details of learning
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
checkpoint_callback = ModelCheckpoint(
    'D:/GITHUB/ComputerVision/AlexNet/210513/flower_weights.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True, mode='min')

reduce_lr_on_plateau = ReduceLROnPlateau(
    monitor='val_accuracy',  # 평탄해지면 lr을 바꿔서 학습할 수 있도록
    factor=0.1,     # 얼마만큼 작아질지(0.1씩 줄어듦)
    patience=5,     # 얼마나 평탄하면 적용?
    min_lr=0.00001,     # 어디까지 작아질지
    verbose=1, mode='min')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=2162//128,
    validation_data=valid_geneator,
    epochs=100,
    callbacks=[early_stopping, checkpoint_callback, reduce_lr_on_plateau]
)

fig, loss_acc = plt.subplots()
acc = loss_acc.twinx()

loss_acc.plot(history.history['loss'], 'y', label='train loss')
loss_acc.plot(history.history['val_loss'], 'r', label='val loss')
loss_acc.set_xlabel('epoch')
loss_acc.set_ylabel('loss')
loss_acc.legend(loc='upper left')

acc.plot(history.history['accuracy'], 'b', label='train acc')
acc.plot(history.history['val_accuracy'], 'g', label='val acc')
acc.set_ylabel('accuracy')
acc.legend(loc='lower left')

plt.show()


scores = model.evaluate_generator(generator=test_generator)
print("Total Loss : ", scores[0], "\nTotal Accuracy : ", scores[1])

## 결과를 시각화
image_path = 'D:/GITHUB/ComputerVision/AlexNet/210513/test/sunflower/9610373158_5250bce6ac_n.jpg'
image = cv2.imread(image_path)
output = image.copy()
output = cv2.resize(output, (227, 227))
output = output.astype('float')/225.0
output = np.array(output)
output = np.expand_dims(output, axis=0)

probability = model.predict(output)[0]
idx = np.argmax(probability)

print(probability, idx)

# from google.colab.patches import cv2_imshow

# label = "{}".format(labels[idx])
# cv2.putText(image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

cv2.imshow('result', image)