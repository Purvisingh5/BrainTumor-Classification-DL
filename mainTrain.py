import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical

image_directory='brain_tumor_dataset/'

no_tumor_images=os.listdir(image_directory+ 'no/')
yes_tumor_images=os.listdir(image_directory+ 'yes/')
dataset=[]
label=[]

INPUT_SIZE=120
# print(no_tumor_images)

# path='no0.jpg'

# print(path.split('.')[1])

for i , image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'no/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i , image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'yes/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset=np.array(dataset)
label=np.array(label)


x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.2, random_state=0)

# Reshape = (n, image_width, image_height, n_channel)

# print(x_train.shape)
# print(y_train.shape)

# print(x_test.shape)
# print(y_test.shape)

x_train=normalize(x_train, axis=1)
x_test=normalize(x_test, axis=1)

y_train=to_categorical(y_train , num_classes=2)
y_test=to_categorical(y_test , num_classes=2)



# Model Building
# 120,120,3

model=Sequential()

model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))


# Binary CrossEntropy= 1, sigmoid
# Categorical Cross Entryopy= 2 , softmax

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])


model.fit(x_train, y_train, 
batch_size=16, 
verbose=1, epochs=10, 
validation_data=(x_test, y_test),
shuffle=False)
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#  augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Initialize lists to store performance per fold
accuracies = []
precisions = []
recalls = []
f1_scores = []
losses = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(dataset):
  # Split data into training and testing sets
  x_train_fold, x_test_fold = dataset[train_index], dataset[test_index]
  y_train_fold, y_test_fold = label[train_index], label[test_index]

  # Normalize data (if necessary)
  x_train_fold = normalize(x_train_fold, axis=1)
  x_test_fold = normalize(x_test_fold, axis=1)

  # One-hot encode labels
  y_train_fold = to_categorical(y_train_fold, num_classes=2)
  y_test_fold = to_categorical(y_test_fold, num_classes=2)

  # Data augmentation for training data only
  train_datagen = datagen.flow(x_train_fold, y_train_fold, batch_size=18)

  # Train the model with augmented data
  model.fit(train_datagen, epochs=8, batch_size=16  ,validation_data=(x_test_fold, y_test_fold))

  # Evaluate the model on the test fold (no augmentation here)
  test_loss, test_acc = model.evaluate(x_test_fold, y_test_fold, verbose=0)

  # Storing the metrics
  losses.append(test_loss)
  accuracies.append(test_acc)

  # Get predictions and calculate precision, recall, F1-score
  y_pred = model.predict(x_test_fold)
  y_pred_classes = np.argmax(y_pred, axis=1)
  y_true_classes = np.argmax(y_test_fold, axis=1)

  precision = precision_score(y_true_classes, y_pred_classes)
  recall = recall_score(y_true_classes, y_pred_classes)
  f1 = f1_score(y_true_classes, y_pred_classes)

  precisions.append(precision)
  recalls.append(recall)
  f1_scores.append(f1)

model.save('BrainTumorCategorical.h5')





