import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

correction = 0.1

lines = []
with open('training_data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

images = []
measurements = []
for line in lines:
  source_path = line[0]
  filename = source_path.split('\\')[-1]
  #print(filename[-1])
  current_path = 'training_data/IMG/' + filename
  #print(current_path)
  image = cv2.imread(current_path)
  images.append(image)
  measurement = float(line[3])
  measurements.append(measurement)
  # Also add a flipped image
  images.append(cv2.flip(image,1))
  measurements.append(measurement*-1)

  # Add the left Images to the training set
  #source_path = line[1]
  #filename = source_path.split('\\')[-1]
  ##print(filename[-1])
  #current_path = 'training_data/IMG/' + filename
  ##print(current_path)
  #image = cv2.imread(current_path)
  #images.append(image)
  #measurement = float(line[3]) + correction
  #measurements.append(measurement)
  #
  ## Add the right images to the training set
  #source_path = line[2]
  #filename = source_path.split('\\')[-1]
  ##print(filename[-1])
  #current_path = 'training_data/IMG/' + filename
  ##print(current_path)
  #image = cv2.imread(current_path)
  #images.append(image)
  #measurement = float(line[3]) - correction
  #measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping= ((60,23),(0,0))))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')

