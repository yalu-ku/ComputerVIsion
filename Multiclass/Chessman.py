import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import utils
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


## 폴더가 없으면 새로운 폴더를 생성
def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


## 데이터 나누기
root_path = os.getcwd() + "/Chessman-image-dataset"
train_path = root_path + "/train"
valid_path = root_path + "/valid"
data_path = root_path + "/Chess"

make_dir(train_path)  # root path 안에 없으면 만들어라
make_dir(valid_path)

class_list = os.listdir(data_path)
class_list.sort()

for class_name in class_list:
    temp_path = train_path + "/" + class_name
    make_dir(temp_path)

for class_name in class_list:
    temp_path = valid_path + "/" + class_name
    make_dir(temp_path)

## imp shutil
for class_name in class_list:
    image_list = os.listdir(data_path + '/' + class_name)
    # 아래부분 수정, 학습 결과
    image_num = len(image_list)
    print(class_name + " folder has " + str(image_num) + " images.")

    train_index = int(image_num * 0.7)  # 클래스를 7:3의 비율로 나눔
    print(train_index)
    for image in image_list[:train_index]:  # 간단, 바이너리 참고
        shutil.copy(data_path + "/" + class_name + "/" + image,
                    train_path + "/" + class_name + "/" + image)

    for image in image_list[train_index:]:
        shutil.copy(data_path + "/" + class_name + "/" + image,
                    valid_path + "/" + class_name + "/" + image)

    print("Image moved.")

## 이미지 개수 파악
for class_name in class_list:
    count = 0
    for image in os.listdir(train_path + '/' + class_name):
        if image[-3:] == "jpg" or image[-3:] == "png" or image[-4:] == "JPEG" or image[-4:] == "jpeg":
            count += 1
    print("Train " + class_name + " : " + str(count))

    count = 0
    for image in os.listdir(valid_path + '/' + class_name):
        if image[-3:] == "jpg" or image[-3:] == "png" or image[-4:] == "JPEG" or image[-4:] == "jpeg":
            count += 1
    print("Valid " + class_name + " : " + str(count) + '\n')


# # 0아니면 1로 라벨을 설정하는 코드
# for class_name in class_list:
#   count = 0
#   for image in os.listdir(train_path + '/' + class_name):
#     if image[-3:] == "jpg" or image[-3:] == "png" or image[-4:] == "JPEG" or image[-4:] == "jpeg":
#       count += 1
#   print("Train " + class_name + " : " + str(count))
#
#   count = 0
#   for image in os.listdir(valid_path + '/' + class_name):
#     if image[-3:] == "jpg" or image[-3:] == "png" or image[-4:] == "JPEG" or image[-4:] == "jpeg":
#       count += 1
#   print("Valid " + class_name + " : " + str(count) + '\n')


## 손수 라벨링해주던 이전의 코드입니다. 다음 셀과 비교해서 봐주세요.
'''
train_data = []
train_label = []
class_index = 0

for class_name in class_list:
    for image in os.listdir(train_path + "/" + class_name):
        if image[-3:] == "jpg" or image[-3:] == "png":
            temp_path = train_path + "/" + class_name + "/"
            subimage = cv2.imread(temp_path + image)
            subimage = cv2.resize(subimage, (200, 200))
            subimage = np.array(subimage) / 255
            train_data.append(subimage)
            train_label.append(class_index)

    class_index += 1

X_train = np.array(train_data)
y_train = np.array(train_label)
print(np.shape(X_train))
print(np.shape(y_train), y_train)

valid_data = []
valid_label = []
class_index = 0

for class_name in class_list:
    for image in os.listdir(valid_path + "/" + class_name):
        if image[-3:] == "jpg" or image[-3:] == "png":
            temp_path = valid_path + "/" + class_name + "/"
            subimage = cv2.imread(temp_path + image)
            subimage = cv2.resize(subimage, (200, 200))
            subimage = np.array(subimage) / 255
            valid_data.append(subimage)
            valid_label.append(class_index)

    class_index += 1

X_valid = np.array(valid_data)
y_valid = np.array(valid_label)
print(np.shape(X_valid))
print(np.shape(y_valid), y_valid)

## 배열의 형태로 만들어줌
y_train = utils.to_categorical(y_train, num_classes=6)
y_valid = utils.to_categorical(y_valid, num_classes=6)

print(y_valid)
'''


## Image Generator
data_generator = ImageDataGenerator(
    rotation_range=40, #이미지를 정해진 값으로 돌리는 연산
    width_shift_range=0.2, #가로로 이미지가 이동하는 비율
    height_shift_range=0.2, #세로로 ~
    shear_range=0.2, #이미지가 기울어진거
    zoom_range=0.2, #얼마나 확대 축소, 범위 지정
    horizontal_flip=True, #수평방향으로 뒤집는
    vertical_flip=True, #수직방향으로 뒤집어서 양을 늘리는
    fill_mode='nearest') #빈공간 어떻게 채울지, 움직이기 이전의 값들을 쭉 늘어뜨려서 가져옴

tmp_path = os.getcwd() + '/temp'
make_dir(tmp_path)
tmp_img = cv2.imread(train_path + "/Bishop/00000001.jpg")
tmp_img = np.array(tmp_img)
tmp_img = tmp_img.reshape((1,) + tmp_img.shape)

count = 0
for batch in data_generator.flow(tmp_img, batch_size=1, save_to_dir=tmp_path):
    if count > 10:
        break
    count += 1

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.4,
    height_shift_range=0.3,
    shear_range=0.35,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True)

valid_datagen = ImageDataGenerator(rescale=1./255) #연산속도 /255 정도만

test_datagen = ImageDataGenerator(rescale=1./255)

# flow_from_directory : 많은 양의 데이터를 효율적으로 가져오는
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(512, 512),
    batch_size = 32, #보통 2의배수로
    class_mode='categorical') #binary

valid_generator = valid_datagen.flow_from_directory(
    valid_path,
    target_size=(512, 512),
    batch_size = 32,
    class_mode='categorical')

# test set은 새로운 이미지로
test_generator = test_datagen.flow_from_directory(
    valid_path,
    target_size=(512,512),
    batch_size = 32,
    class_mode='categorical')


## 지정된 위치 따로 없음 [하이퍼파라미터]
def vanilla_model():
    input = Input(shape=(512, 512, 3))
    conv1 = Conv2D(16, (3, 3), padding='same', activation='relu')(input)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    batch1 = BatchNormalization()(pool1)

    conv2 = Conv2D(32, (3, 3), padding='same', activation='relu')(batch1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    batch2 = BatchNormalization()(pool2)

    conv3 = Conv2D(64, (3, 3), padding='same', activation='relu')(batch2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    batch3 = BatchNormalization()(pool3)

    conv4 = Conv2D(128, (3, 3), padding='same', activation='relu')(batch3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    batch4 = BatchNormalization()(pool4)

    flatten = Flatten()(pool4)
    dense1 = Dense(256, activation='relu')(flatten)
    drop1 = Dropout(0.3)(dense1)
    dense2 = Dense(64, activation='relu')(drop1)
    drop2 = Dropout(0.4)(dense2)
    output = Dense(6, activation='softmax')(drop2) # class 갯수만큼 node 수 설정
    # softmax : 각각의 class에 대해서 score(중요도) + 가장 높은거 출력

    model = Model(inputs=input, outputs=output)

    return model

#
# def vanilla_model():  # sequenc
#     input = Input(shape=(200, 200, 3))
#     conv1 = Conv2D(16, (3, 3), padding='same', activation='relu')(input)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     conv2 = Conv2D(32, (3, 3), padding='same', activation='relu')(pool1)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#     conv3 = Conv2D(64, (3, 3), padding='same', activation='relu')(pool2)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#     conv4 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool1)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#     flatten = Flatten()(pool4)
#     dense1 = Dense(256, activation='relu')(flatten)
#     output = Dense(6, activation='softmax')(dense1)
#
#     model = Model(inputs=input, outputs=output)
#
#     return model
#

model = vanilla_model()
opt = Adam(lr=0.001, decay=1e-6)  # (learning rate, 수렴에 대해 알려주는)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print(model.summary())
# <-> loss='binary-'

## 여기까지 네트워크, 다음은 학습을 시켜야 함

#
# # ~.fit : 학습시킴
## 이미지의 배열과 라벨을 따로 ...
'''
# history = model.fit(X_train, y_train,  # (영상데이터, 라벨데이터,
#                     validation_data=(X_valid, y_valid),  # 튜플형태
#                     batch_size=32,  # 이미지 몇개로 나눠서
#                     epochs=10, verbose=1)
#

# ## val 성능 테스트
# scores = model.evaluate_generator(generator=test_generator)
# print("Total Loss : ", scores[0], "\nTotal Accuracy : ", scores[1])
'''

## ImageDataGenerator로 받은 이미지를 학습하는 부분
early_stopping = EarlyStopping(monitor='val_loss', patience=3) #(loss가 줄어드는 것을 확인, 인내심?)
checkpoint_callback = ModelCheckpoint('multiclass_weight.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit(
    train_generator, #통째로 넣어줌
    steps_per_epoch=370//32, #epoch한번에 step얼마나 /batch사이즈(위에서 32로 지정했음)
    validation_data=valid_generator,
    epochs=50,
    callbacks=[early_stopping, checkpoint_callback] #위에서 설정한 부분
)

## 그래프
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

## 성능 테스트
scores = model.evaluate(test_generator)
print("Total Loss: ", scores[0], "\nTotal Accuracy: ", scores[1])

