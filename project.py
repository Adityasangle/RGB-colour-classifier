#importing all libraries

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
from os import listdir
from numpy import asarray,save
from keras.preprocessing.image import load_img,img_to_array
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
from termcolor import colored

#labelling the photos for dataset


def label_dataset():
    folder = 'DATA/Train/'
    photos, labels = list(), list()
    # enumerate files in the directory
    for file in listdir(folder):
        # determine class
        output = 0.0
        if file.startswith('red'):
            output = 1.0
        if file.startswith('green'):
            output = 2.0
        if file.startswith('blue'):
            output = 3.0	
        # load image
        photo = load_img(folder + file, target_size=(80, 80))        
        # convert to numpy array
        photo = img_to_array(photo)
        # store
        photos.append(photo)
        labels.append(output)
    # convert to a numpy arrays
    photos = asarray(photos)
    labels = asarray(labels)
    save('photos.npy', photos)
    save('labels.npy', labels)
    
    return photos,labels
def label_testdata():
    folder = 'DATA/test/'
    test_photos, test_labels = list(), list()
    # enumerate files in the directory
    for file in listdir(folder):
        # determine class
        output = 0.0
        if file.startswith('red'):
            output = 1.0
        if file.startswith('green'):
            output = 2.0
        if file.startswith('blue'):
            output = 3.0	
        # load image
        photo = load_img(folder + file, target_size=(80,80))        
        # convert to numpy array
        photo = img_to_array(photo)
        # store
        test_photos.append(photo)
        test_labels.append(output)
    # convert to a numpy arrays
    test_photos = asarray(test_photos)
    test_labels = asarray(test_labels)
    #test_labels=test_labels.reshape(1,18)
    #print(test_photos.shape, test_labels.shape)
    # save the reshaped photos
    save('photos.npy', test_photos)
    save('labels.npy', test_labels)
    return test_photos,test_labels
def prep_pixels(train):
    #convert from integers to floats
	train_norm = train.astype('float32')
	
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	
	# return normalized images````
	return train_norm
def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (1, 1), activation='relu', input_shape=photos.shape[1:]))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (1, 1), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='sigmoid'))
    return model

def labelling(output):
  output=output*100
  if (output[0][1]>1):
   print(colored("RED","red"))
  elif (output[0][2]>1):
    print(colored("GREEN","green"))
  else:
    print(colored("BLUE","blue"))
    
def preprocess(img):
  photo=load_img(img,target_size=(80,80))
  photo = img_to_array(photo)
  print('PIXEL VALUES :','RED:',photo[0][0][0],'GREEN:',photo[0][0][1],'BLUE:',photo[0][0][2])
  photo= photo.astype('float32')
  photo=prep_pixels(photo)
  output=model.predict(photo.reshape(-1,80,80,3))
  labelling(output)
  plt.imshow(photo)


photos,labels=label_dataset()
test_photo,test_label=label_testdata()
labels=labels.reshape(42,1)
encoded = to_categorical(labels)

photos=prep_pixels(photos)
test_photos=prep_pixels(test_photo)

model=cnn_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(photos, encoded, epochs=10,batch_size=1)


img=input('Enter path of image(without quotations) : ')
preprocess(img)

