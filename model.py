import os
import csv
from random import shuffle

samples = []
with open('./training_data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Global variables
use_side = True
use_flipped = True
correction = 0.3

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './training_data2/IMG/'+batch_sample[0].split('\\')[-1]
                #print(name)
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                if use_side:
                    left = './training_data2/IMG/'+batch_sample[1].split('\\')[-1]
                    right = './training_data2/IMG/'+batch_sample[2].split('\\')[-1]
                    left_img = cv2.imread(left)
                    right_img = cv2.imread(right)
                    images.append(left_img)
                    images.append(right_img)
                    #images.extend(left_img, right_img)
                    left_angle = center_angle + correction
                    right_angle = center_angle - correction
                    angles.append(left_angle)
                    angles.append(right_angle)
                    #angles.extend(left_angle, right_angle)
                if use_flipped:
                    images.append(cv2.flip(center_image,1))
                    angles.append(center_angle*-1.0)
                    
                     

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=( row, col, ch),
        output_shape=(row, col, ch)))
model.add(Cropping2D(cropping= ((60,23),(0,0))))
#model.add(Convolution2D(6,5,5,activation='relu'))
#model.add(MaxPooling2D())
#model.add(Convolution2D(6,5,5,activation='relu'))
#model.add(MaxPooling2D())
#model.add(Flatten())
#model.add(Dense(120))
#model.add(Dense(84))
#model.add(Dense(1)u)

#NVIDIA BASED MODEL
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
#model.add(Dense(300))
#model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)
#model.fit_generator(train_generator, steps_per_epoch= len(train_samples), validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)
model.save('model.h5')

print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
#plt.set_ylim([0,0.2])
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('training_history.png')


