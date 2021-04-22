## 2018270979 윤아로

import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import utils
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta, Nadam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


## Make dir. and Separate data
def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


root_path = os.getcwd() + "/flowers-dataset"
train_path = root_path + "/train"
valid_path = root_path + "/valid"
test_path = root_path + "/test"
data_path = root_path + "/flowers"

## test용 주석
make_dir(train_path)
make_dir(valid_path)
make_dir(test_path)

class_list = os.listdir(data_path)
class_list.sort()

for class_name in class_list:
    temp_path = train_path + "/" + class_name
    make_dir(temp_path)

for class_name in class_list:
    temp_path = valid_path + "/" + class_name
    make_dir(temp_path)

for class_name in class_list:
    temp_path = test_path + "/" + class_name
    make_dir(temp_path)

for class_name in class_list:
    image_list = os.listdir(data_path + '/' + class_name)
    image_num = len(image_list)
    print(class_name + " folder has " + str(image_num) + " images.")

    train_index = int(image_num * 0.8)
    valid_index = int(image_num * 0.9)
    print("train_index : ", train_index, "\nvalid_index : ", valid_index)

    for image in image_list[:train_index]:
        shutil.copy(data_path + "/" + class_name + "/" + image,
                    train_path + "/" + class_name + "/" + image)

    for image in image_list[train_index:valid_index]:
        shutil.copy(data_path + "/" + class_name + "/" + image,
                    valid_path + "/" + class_name + "/" + image)

    for image in image_list[valid_index:]:
        shutil.copy(data_path + "/" + class_name + "/" + image,
                    test_path + "/" + class_name + "/" + image)

    print("Image moved.")

## Check how many images
for class_name in class_list:
    count = 0
    for image in os.listdir(train_path + '/' + class_name):
        if image[-3:] == "jpg":
            count += 1
    print("Train " + class_name + " : " + str(count))

    count = 0
    for image in os.listdir(valid_path + '/' + class_name):
        if image[-3:] == "jpg":
            count += 1
    print("Valid " + class_name + " : " + str(count))

    count = 0
    for image in os.listdir(test_path + '/' + class_name):
        if image[-3:] == "jpg":
            count += 1
    print("Test " + class_name + " : " + str(count) + '\n')

## Self Labeling
train_data = []
train_label = []
class_index = 0

for class_name in class_list:
    for image in os.listdir(train_path + '/' + class_name):
        if image[-3:] == "jpg":  ## error handling
            temp_path = train_path + '/' + class_name + '/'
            subimage = cv2.imread(temp_path + image)
            subimage = cv2.resize(subimage, (200, 200))
            subimage = np.array(subimage) / 255
            train_data.append(subimage)
            train_label.append(class_index)
    class_index += 1

X_train = np.array(train_data)
y_train = np.array(train_label)
# print(np.shape(X_train))
# print(np.shape(y_train), y_train)

valid_data = []
valid_label = []
class_index = 0

for class_name in class_list:
    for image in os.listdir(valid_path + '/' + class_name):
        if image[-3:] == "jpg":
            temp_path = valid_path + '/' + class_name + '/'
            subimage = cv2.imread(temp_path + image)
            subimage = cv2.resize(subimage, (200, 200))
            subimage = np.array(subimage) / 255
            valid_data.append(subimage)
            valid_label.append(class_index)
    class_index += 1

X_valid = np.array(valid_data)
y_valid = np.array(valid_label)
# print(np.shape(X_valid))
# print(np.shape(y_valid), y_valid)

test_data = []
test_label = []
class_index = 0

for class_name in class_list:
    for image in os.listdir(test_path + '/' + class_name):
        if image[-3:] == "jpg":
            temp_path = test_path + '/' + class_name + '/'
            subimage = cv2.imread(temp_path + image)
            subimage = cv2.resize(subimage, (200, 200))
            subimage = np.array(subimage) / 255
            test_data.append(subimage)
            test_label.append(class_index)
    class_index += 1

X_test = np.array(test_data)
y_test = np.array(test_label)
# print(np.shape(X_test))
# print(np.shape(y_test), y_test)

## Make an array type
y_train = utils.to_categorical(y_train, num_classes=5)
y_valid = utils.to_categorical(y_valid, num_classes=5)
y_test = utils.to_categorical(y_test, num_classes=5)

'''
## Image Generator
data_generator = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

tmp_path = os.getcwd() + '/temp'
make_dir(tmp_path)
tmp_img = cv2.imread(train_path + "/daisy/21652746_cc379e0eea_m.jpg")
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
    vertical_flip=True
    )
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(200,200),
    batch_size=32,
    class_mode='categorical'
)
valid_generator = valid_datagen.flow_from_directory(
    valid_path,
    target_size=(200,200),
    batch_size=32,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(200,200),
    batch_size=32,
    class_mode='categorical'
)
'''
def vanila_model():
    input = Input(shape=(200, 200, 3))
    conv1 = Conv2D(8, (3, 3), padding='same', activation='relu')(input)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    batch1 = BatchNormalization()(pool1)

    conv2 = Conv2D(16, (3, 3), padding='same', activation='relu')(batch1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    batch2 = BatchNormalization()(pool2)

    conv3 = Conv2D(32, (3, 3), padding='same', activation='relu')(batch2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    batch3 = BatchNormalization()(pool3)

    conv4 = Conv2D(64, (3, 3), padding='same', activation='relu')(batch3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    batch4 = BatchNormalization()(pool4)

    conv5 = Conv2D(128, (3, 3), padding='same', activation='relu')(batch4)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    batch5 = BatchNormalization()(pool5)

    conv6 = Conv2D(256, (3, 3), padding='same', activation='relu')(batch5)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)
    batch6 = BatchNormalization()(pool6)

    # conv7 = Conv2D(1024, (3, 3), padding='same', activation='relu')(batch6)
    # pool7 = MaxPooling2D(pool_size=(2, 2))(conv7)
    # batch7 = BatchNormalization()(pool7)

    ## 레이어 개수 최대
    # conv8 = Conv2D(1024, (3, 3), padding='same', activation='relu')(batch7)
    # pool8 = MaxPooling2D(pool_size=(2, 2))(conv8)
    # batch8 = BatchNormalization()(pool8)

    flatten = Flatten()(batch6)
    dense1 = Dense(256, activation='relu')(flatten)
    drop1 = Dropout(0.3)(dense1)
    # dense2 = Dense(128, activation='relu')(drop1)
    # drop2 = Dropout(0.3)(dense2)
    output = Dense(5, activation='softmax')(drop1)

    model = Model(inputs=input, outputs=output)

    return model


model = vanila_model()
# opt = Adam(lr=0.001, decay=1e-6)
opt2 = RMSprop(lr=0.001, decay=1e-6)
# opt3 = Adadelta(lr=0.001, decay=1e-6)
opt4 = Nadam(lr=0.001, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=opt4, metrics=['accuracy'])
print(model.summary())

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
checkpoint_callback = ModelCheckpoint('multiclass_weight.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                      mode='min')

## Self labeling model fit
history = model.fit(
    X_train, y_train,
    # steps_per_epoch=2594 // 128,
    validation_data=(X_valid, y_valid),
    epochs=40,
    callbacks=[early_stopping, checkpoint_callback]
)

## Image Generator modle fit
# history = model.fit(
#     train_generator,
#     # steps_per_epoch=2594//128,
#     validation_data=valid_generator,
#     epochs=50,
#     callbacks=[early_stopping, checkpoint_callback]
# )


## Graph
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


## Test
scores = model.evaluate(X_test, y_test)
# scores = model.evaluate(test_generator)
print("Total Loss: ", scores[0], "\nTotal Accuracy: ", scores[1])

print(model.predict(X_test))
