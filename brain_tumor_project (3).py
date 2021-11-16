# -*- coding: utf-8 -*-
"""Brain_tumor_project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1G6JT9E-yz-xDH-Hxe41dckU-yNvsteuo

### Import librabries
"""

from google.colab import drive
drive.mount('/content/drive')

pip install Keras-Applications

import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
from tqdm.notebook import tqdm

from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, ConfusionMatrixDisplay
import keras
from keras.layers import Dense, Conv2D, Activation,BatchNormalization, MaxPooling2D, Dropout, Flatten, Input
from keras.models import load_model, Sequential
from keras.layers.advanced_activations import LeakyReLU

#import cv2

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

"""### 1. Read the dataset

#### 1.2 Read labels
"""

# read labels
df_labels = pd.read_csv('/content/drive/MyDrive/data/Brain/dataset/dataset/label.csv')

df_labels.head(10)

df_labels.count()

df_labels['tumor'] = 1
df_labels['tumor'][df_labels['label'] == 'no_tumor'] = 0

plt.figure()
# To change the color visit: https://seaborn.pydata.org/tutorial/color_palettes.html?highlight=color 
sns.color_palette("muted")  
sns.countplot(x='tumor',  data=df_labels)
plt.xticks([0,1], ['No tumor', 'tumor'])
plt.xlabel("")
plt.show()

df_labels['tumor_type'] = df_labels['tumor']
df_labels['tumor_type'] = df_labels['label'].map({'no_tumor': 0, 'meningioma_tumor': 1, 'glioma_tumor': 2, 'pituitary_tumor': 3 })

plt.figure()
sns.countplot(x='tumor_type',  data=df_labels)
plt.xticks([0,1,2,3], ['No tumor', 'meningioma','glioma', 'pituitary'])
plt.xlabel("")
plt.title("Tumor types")
plt.show()

df_labels.to_csv('p_labels.csv')

"""#### 2.2 Read images"""

df_labels.head()

Img_id = df_labels['file_name']

# "grayscale" for baseline models  "rgb" for cnn

imgs =[np.array(image.load_img('/content/drive/MyDrive/data/Brain/dataset/dataset/image/{}'.format(i),target_size=(100,100), color_mode = "rgb"))/255 for i in tqdm(Img_id[:Img_id.size])]

imgs_arr = np.array(imgs)

imgs_arr.shape

# show samples
plt.figure(figsize=(15, 10))
for i in range(6):
    plt.subplot(1, 6, i+1)
    plt.imshow(imgs_arr[i],cmap='gray')
    plt.title(df_labels.label[i])
    plt.axis('off')
    
plt.tight_layout()
plt.show()

# for grayscale only
imgs_arr = imgs_arr.reshape(imgs_arr.shape[0], imgs_arr.shape[1], imgs_arr.shape[2], 1)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(imgs_arr, df_labels.tumor, test_size=0.2, random_state=42, stratify=df_labels.tumor)

"""### 2. Baseline models"""

X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]*X_train.shape[3]))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]*X_test.shape[3]))

"""#### 2.1 KNN"""

from sklearn.neighbors import KNeighborsClassifier

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the training data
with tf.device('/gpu:0'):
  knn.fit(X_train_reshaped, y_train)

# Print the accuracy
print("Training accuracy: ", knn.score(X_train_reshaped, y_train))

print("Test accuracy: ", knn.score(X_test_reshaped, y_test))

y_pred = knn.predict(X_test_reshaped)
target_names = ['No tumor', 'Tumor']
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

plot_confusion_matrix(knn, X_test_reshaped, y_test, normalize = "true", cmap=plt.cm.Blues, display_labels=['Not tumor', 'Tumor'])  
plt.show()

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors = k)

    # Fit the classifier to the training data
    with tf.device('/gpu:0'):
      knn.fit(X_train_reshaped, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train_reshaped, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test_reshaped, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

"""####2.2 SVM"""

# SVM
from sklearn.svm import SVC

# Instantiate the SVC classifier: clf
clf = SVC()

# Fit the pipeline to the train set
with tf.device('/gpu:0'):
  clf.fit(X_train_reshaped, y_train)

# Print the accuracy
print("Training accuracy: ", knn.score(X_train_reshaped, y_train))

print("Test accuracy: ", knn.score(X_test_reshaped, y_test))

# Predict the labels of the test set
y_pred = clf.predict(X_test_reshaped)

# Compute metrics
target_names = ['No tumor', 'Tumor']
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

# normalized
plot_confusion_matrix(clf, X_test_reshaped, y_test, normalize = "true", cmap=plt.cm.Blues, display_labels=['Not tumor', 'Tumor'])  
plt.show()

plot_confusion_matrix(clf, X_test_reshaped, y_test, cmap=plt.cm.Blues, values_format=".0f", display_labels=['Not tumor', 'Tumor'])  
plt.show()

"""### 3. Main model"""

# Things to do:  1. Cross validation

# cross-validation

from sklearn.model_selection import cross_val_score

cv_results = cross_val_score(reg, X, y, cv=5)
print(cv_results)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_results)))

"""#### 3.1 Handling data imbalance (class weights)

##### CNN ( good)
"""

## CNN 

model = Sequential()
model.add(Conv2D(64, (5,5), input_shape = (X_train[0].shape[0], X_train[0].shape[1], 3), padding = 'same'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size =  (2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (5,5), padding = 'same'))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size =  (2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3,3), padding = 'same'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size =  (2,2)))
model.add(Dropout(0.2))


model.add(Flatten())

model.add(Dense(128))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

# Compile the model
model.compile(loss=keras.losses.binary_crossentropy, metrics=['accuracy'], optimizer='adam')

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced'
                                               ,np.unique(y_train)
                                               ,y_train)

computed_class_weights = dict(enumerate(class_weights))

# Create training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

# Fit the model
with tf.device('/gpu:0'):
  history = model.fit(X_train, y_train,
                    epochs = 30,
                    batch_size = 100,
                    validation_data=(X_val,y_val),
                    #class_weight = computed_class_weights,
                    verbose=1)

"""Results with class weights:
Epoch 1/5
24/24 [==============================] - 301s 13s/step - loss: 0.6441 - accuracy: 0.7638 - val_loss: 0.3410 - val_accuracy: 0.8567

Epoch 2/5
24/24 [==============================] - 303s 13s/step - loss: 0.3454 - accuracy: 0.8483 - val_loss: 0.4639 - val_accuracy: 0.8483

Epoch 3/5
24/24 [==============================] - 295s 12s/step - loss: 0.2820 - accuracy: 0.8892 - val_loss: 0.4256 - val_accuracy: 0.8483

Epoch 4/5
24/24 [==============================] - 302s 13s/step - loss: 0.2352 - accuracy: 0.9112 - val_loss: 0.5433 - val_accuracy: 0.8483

Epoch 5/5
24/24 [==============================] - 298s 12s/step - loss: 0.2024 - accuracy: 0.9137 - val_loss: 0.4604 - val_accuracy: 0.8483
"""

# Utility function for plotting of the model results
def visualize_results(history):
    # Plot the accuracy and loss curves
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

# Run the function to illustrate accuracy and loss
visualize_results(history)

from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
target_names = ['No tumor', 'Tumor']
print(classification_report(y_test, y_pred.round(), target_names=target_names, digits=4))

# Not Normalized
cm = confusion_matrix(y_test, y_pred.round())
print(cm)

# Normalized
cm = confusion_matrix(y_test, y_pred.round(), normalize='true')
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not tumor', 'Tumor'])
disp = disp.plot(cmap=plt.cm.Blues)
plt.show()

"""##### Fine-tuning"""

from keras.models import Model

#from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
#from keras.applications.resnet50 import ResNet50
#from tensorflow.keras.applications import ResNet50


# load model and specify a new input shape for images
input_tensor = Input(shape=(100, 100, 3))

# creating the base model of pre-trained VGG16 model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

# Freeze all the layers
for layer in base_model.layers[:]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in base_model.layers:
    print(layer, layer.trainable)

# build a classifier model to put on top of the convolutional model
model = Sequential()
model.add(base_model)
model.add(Flatten(input_shape=base_model.output_shape[1:]))

model.add(Dense(256))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# defining a function to save the weights of best model
#from keras.callbacks import ModelCheckpoint
#mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')

# Compile the model
model.compile(loss=keras.losses.binary_crossentropy, metrics=['accuracy'], optimizer='adam')

# Create training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced'
                                               ,np.unique(y_train)
                                               ,y_train)

computed_class_weights = dict(enumerate(class_weights))

computed_class_weights

# Fit the model

history = model.fit(X_train, y_train,
                    epochs = 30,
                    batch_size = 100,
                    validation_data=(X_val,y_val),
                    class_weight= computed_class_weights)

# Run the function to illustrate accuracy and loss
visualize_results(history)

y_pred = model.predict(X_test)
target_names = ['No tumor', 'Tumor']
print(classification_report(y_test, y_pred.round(), target_names=target_names, digits=4))

# Not Normalized
cm = confusion_matrix(y_test, y_pred.round())
print(cm)

# Normalized
cm = confusion_matrix(y_test, y_pred.round(), normalize='true')
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not tumor', 'Tumor'])
disp = disp.plot(cmap=plt.cm.Blues)
plt.show()

"""##### Fine-tuning with data augmentation"""





# Train images
x_train = []
for i in tqdm(train_x):
    image_path = '/content/drive/MyDrive/data/Brain/dataset/dataset/image/'+i
    img = np.array(image.load_img(image_path, target_size=(100,100), color_mode = "rgb"), dtype="float")/ 255.0
    x_train.append(img)

# Train df
df_train = pd.DataFrame(columns=['file_name','tumor'])
df_train['file_name'] = train_x
df_train['tumor'] = train_y

# Test df
df_test= pd.DataFrame(columns=['file_name','tumor'])
df_test['file_name'] = val_x
df_test['tumor'] = val_y

df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

# Images
train_images = df_train.loc[:,'file_name']
train_labels = df_train.loc[:,'tumor']

test_images = df_test.loc[:,'file_name']
test_labels = df_test.loc[:,'tumor']

# Train images
x_train = []
for i in tqdm(train_images):
    image_path = '/content/drive/MyDrive/data/Brain/dataset/dataset/image/'+i
    img = np.array(image.load_img(image_path, target_size=(100,100), color_mode = "rgb"), dtype="float")/ 255.0
    x_train.append(img)

# Train labels
#y_train=keras.utils.np_utils.to_categorical(train_labels)

# Test images
x_test = []
for i in tqdm(test_images):
    image_path = '/content/drive/MyDrive/data/Brain/dataset/dataset/image/'+i
    img = np.array(image.load_img(image_path, target_size=(100,100), color_mode = "rgb"), dtype="float")/ 255.0
    x_test.append(img)

# Test labels
#y_test=keras.utils.np_utils.to_categorical(test_labels)

x_train = np.array(x_train)
x_test = np.array(x_test)

#x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)

#x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

from keras.models import Model

from keras.applications.vgg16 import VGG16
#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.resnet50 import ResNet50

# load model and specify a new input shape for images
input_tensor = Input(shape=(100, 100, 3))

# creating the base model of pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

# Freeze all the layers
for layer in base_model.layers[:]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in base_model.layers:
    print(layer, layer.trainable)

# build a classifier model to put on top of the convolutional model
model = Sequential()
model.add(base_model)
model.add(Flatten(input_shape=base_model.output_shape[1:]))

model.add(Dense(256))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced'
                                               ,np.unique(train_y)
                                               ,train_y)

computed_class_weights = dict(enumerate(class_weights))

df_train.head()

from keras.preprocessing.image import ImageDataGenerator

# Augmentation
train_datagen = ImageDataGenerator(rotation_range=10,  # rotation
                                   #width_shift_range=0.1,  # horizontal shift
                                   # zoom
                                   horizontal_flip=True,  # horizontal flip
                                   vertical_flip=True,
                                   fill_mode="constant"
                                   #brightness_range=[0.2,0.5]
                                   )  # brightness

# ImageDataGenerator flow_from_dataframe

#df_train = pd.read_csv('/content/drive/MyDrive/data/Brain/dataset/dataset/label.csv')
df_train['tumor'] = df_train['tumor'].astype('str')

train_generator = train_datagen.flow_from_dataframe(dataframe=df_train, 
                                              directory='/content/drive/MyDrive/data/Brain/dataset/dataset/image',
                                              x_col="file_name", 
                                              y_col="tumor",                          
                                              class_mode="binary", 
                                              target_size=(100, 100), 
                                              batch_size=1,
                                              rescale=1.0/255,
                                              seed=2020)
# plotting images
fig, ax = plt.subplots(nrows=1, ncols=10, figsize=(15,15))

for i in range(10):

  # convert to unsigned integers for plotting
  img = next(train_generator)[0].astype('uint8')

  # changing size from (1, 100, 100, 3) to (100, 100, 3) for plotting the image
  img = np.squeeze(img)

  # plot raw pixel data
  ax[i].imshow(img)
  ax[i].axis('off')

valid_datagen = ImageDataGenerator(rotation_range=20,  # rotation
                                   #width_shift_range=0.1,  # horizontal shift
                                    # zoom
                                   horizontal_flip=True,  # horizontal flip
                                   vertical_flip=True,
                                   fill_mode="constant"
                                   #brightness_range=[0.2,0.5]
                                   )  # brightness

valid_generator = valid_datagen.flow_from_dataframe(dataframe=df_test, 
                                                    directory='/content/drive/MyDrive/data/Brain/dataset/dataset/image',
                                                    x_col="file_name", 
                                                    y_col="tumor", 
                                                    class_mode="binary", 
                                                    target_size=(100, 100), 
                                                    batch_size=1,
                                                    rescale=1.0/255,
                                                    seed=2020)

# Compile
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Epochs
epochs = 10
# Batch size
batch_size = 100

history = model.fit(train_generator,
                    batch_size=batch_size, 
                    epochs=epochs,
                    validation_data=valid_generator,
                    class_weight = computed_class_weights,
                    verbose=1)

# Run the function to illustrate accuracy and loss
visualize_results(history)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred.round()))

print(confusion_matrix(y_test, y_pred.round()))

model.save("tumor_classification.h5")
print("Saved model to disk")