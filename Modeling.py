import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 현재 위치 확인
print("Now we are here : ", os.getcwd())


# 폴더를 생성하기 위한 함수
def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


# 경로 지정
root_path = os.getcwd()
data_path = root_path + "/dogs"
origin_data_path = data_path + "/OriginData"
new_data_path = data_path + "/newData"
train_path = new_data_path + "/train"
valid_path = new_data_path + "/valid"
test_path = new_data_path + "/test"

# make_dir(new_data_path)
# make_dir(train_path)
# make_dir(valid_path)
# make_dir(test_path)
#
# class_list = os.listdir(origin_data_path)
# # class_list.sort()
#
# for class_name in class_list:
#     temp_path = train_path + '/' + class_name
#     make_dir(temp_path)
#
# for class_name in class_list:
#     temp_path = valid_path + '/' + class_name
#     make_dir(temp_path)
#
# for class_name in class_list:
#     temp_path = test_path + '/' + class_name
#     make_dir(temp_path)
#
# # copy newData from originData
# for class_name in class_list:
#     image_list = os.listdir(origin_data_path + '/' + class_name)
#     image_num = len(image_list)  # 이미지 개수
#     print(class_name + ' folder has ' + str(image_num) + ' images.')
#
#     train_index = int(image_num * 0.8)
#     valid_index = int(train_index * 0.7)
#     for image in image_list[:valid_index]:
#         shutil.copy(origin_data_path + '/' + class_name + '/' + image,
#                     train_path + '/' + class_name + '/' + image)
#     for image in image_list[valid_index:train_index]:
#         shutil.copy(origin_data_path + '/' + class_name + '/' + image,
#                     valid_path + '/' + class_name + '/' + image)
#     for image in image_list[train_index:]:
#         shutil.copy(origin_data_path + '/' + class_name + '/' + image,
#                     test_path + '/' + class_name + '/' + image)
#     print("Image moved.")

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   fill_mode='nearest'
                                   )

valid_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_path,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)


def build_model():
    conv_base = VGG16(include_top=False, input_shape=(150, 150, 3))
    conv_base.trainable = False
    x = conv_base.input
    gap = layers.GlobalAvgPool2D()(conv_base.output)
    dense = layers.Dense(2)(gap)
    batch = layers.BatchNormalization()(dense)
    act = layers.Activation(activation='relu')(batch)
    do = layers.Dropout(0.25)(act)
    y = layers.Dense(1, activation='sigmoid')(do)

    model = models.Model(x, y)
    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0001, momentum=0.4), loss='binary_crossentropy', metrics=['accuracy'])

    return model


# def build_model():
#     model = models.Sequential()
#     conv_base = vgg16.VGG16(include_top=False, input_shape=(150, 150, 3))
#     conv_base.trainable = False
#     model.add(conv_base)
#     # model.add(layers.Flatten())
#     model.add(layers.GlobalAvgPool2D())
#     model.add(layers.Dense(256, activation='relu'))
#     model.add(layers.Dense(1, activation='sigmoid'))
#
#     model.compile(optimizer=optimizers.RMSprop(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
#     return model


model = build_model()
model.summary()
history = model.fit(train_generator, epochs=85, validation_data=valid_generator)


def plot(h, met):
    plt.plot(h.history[met])
    plt.plot(h.history['val_' + met])
    plt.title(met)
    plt.xlabel('epoch')
    plt.ylabel(met)
    plt.legend(['train', 'valid'], loc=0)


plot(history, 'loss')
plt.savefig('VGG16_2_mom_loss_ep.png')
plt.clf()
plot(history, 'accuracy')
plt.savefig('VGG16_2_mom_accuracy_ep.png')

model.save('VGG16_mom_pretrain_2_ep.h5')
