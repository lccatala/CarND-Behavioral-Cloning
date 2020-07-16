import cv2
from scipy import ndimage

######################
## GENERATOR
######################

from sklearn.model_selection import train_test_split

import numpy as np
import sklearn
"""
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
  num_samples = len(samples)
  while 1: # Loop forever so the generator never terminates
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]

      images = []
      angles = []
      for batch_sample in batch_samples:
        name = './IMG/'+batch_sample[0].split('/')[-1]
        center_image = cv2.imread(name)
        center_angle = float(batch_sample[3])
        images.append(center_image)
        angles.append(center_angle)

      # trim image to only see section with road
      X_train = np.array(images)
      y_train = np.array(angles)
      yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
"""
import csv
def load_data():
    lines = []
    with open('../data/driving_log.csv', 'r') as csvfile:
      reader = csv.reader(csvfile)
      for line in reader:
        lines.append(line)

    images = []
    measurements = []
    for line in lines:
      correction = 0.2 # This is a parameter to tune
      steering_center = float(line[3])
      steering_left = steering_center + correction
      steering_right = steering_center - correction

      # Read in images from left, center and right cameras
      source_path_center = line[0]
      source_path_left = line[1]
      source_path_right = line[2]

      filename_center = source_path_center.split('/')[-1]
      filename_left = source_path_left.split('/')[-1]
      filename_right = source_path_right.split('/')[-1]

      path_center = '../data/IMG/' + filename_center
      path_left = '../data/IMG/' + filename_left
      path_right = '../data/IMG/' + filename_right
      images.extend([ndimage.imread(path_center), ndimage.imread(path_left), ndimage.imread(path_right)])
      measurements.extend([steering_center, steering_left, steering_right])

    X_train = np.array(images)
    y_train = np.array(measurements)
    return X_train, y_train

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Conv2D, MaxPooling2D, Dropout

def copied_model():
    model = Sequential()
    # Normalize and mean-center input data
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), padding='valid'))

    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), padding='valid'))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), padding='valid'))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), padding='valid'))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), padding='valid'))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), padding='valid'))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Dense(256))
    model.add(Dense(64))
    model.add(Dense(1))
    
    return model

def udacity_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(160,320,3)))
    model.add(Conv2D(24, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    #model.summary()

    return model

EPOCHS = 5

if __name__ == '__main__':
    print('Loading data...')
    X_train, y_train = load_data()
    print('Done! Training model...')
    filename_model = 'model.h5'
    model = udacity_model()
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=EPOCHS)
    print('Done! Saving model in', filename_model)
    model.save(filename_model)