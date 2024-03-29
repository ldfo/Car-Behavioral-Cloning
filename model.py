import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Cropping2D
import os
import matplotlib.image as mpimg
import cv2

np.random.seed(666)
IM_HEIGHT = 160
IM_WIDTH = 320
IM_CHANNELS = 3

def augment_images(data_dir, center, left, right, steering_angle):
    # Choose an image from left, center or right and adjust steering angle
    choice = np.random.choice(3)
    if choice == 0:
        image = mpimg.imread(os.path.join(data_dir, left.strip()))
        steering_angle += 0.2
    elif choice == 1:
        image = mpimg.imread(os.path.join(data_dir, right.strip()))
        steering_angle -= 0.2
    elif choice ==2:
        image = mpimg.imread(os.path.join(data_dir, center.strip()))

    # make a random flip on the image
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle

    return image, steering_angle

def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    # Generate training image 
    images = np.empty([batch_size, IM_HEIGHT, IM_WIDTH, IM_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            if is_training and np.random.rand() < 0.6:
                # augment data when in training
                image, steering_angle = augment_images(data_dir, center, left, right, steering_angle)
            else:
                # chooses image from center
                image = mpimg.imread(os.path.join(data_dir, center.strip()))
            # add image and steering angle
            images[i] = image
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers

# load data
data_dir = './data'
test_size = .1

data_df = pd.read_csv(os.path.join(data_dir,'driving_log3.csv'))
data_df.columns = ['center','left','right','steering','throttle','break','speed']

X = data_df[['center', 'left', 'right']].values
y = data_df['steering'].values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=666)

# build keras model
INPUT_SHAPE = (IM_HEIGHT, IM_WIDTH, IM_CHANNELS)
keep_prob = .6
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
# based on nvidia's network architecture
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/127.5-1.0))
model.add(Conv2D(24, 5, activation='elu', strides=(2, 2)))
model.add(Conv2D(36, 5, activation='elu', strides=(2, 2)))
model.add(Conv2D(48, 5, activation='elu', strides=(2, 2)))
model.add(Conv2D(64, 3, activation='elu'))
model.add(Conv2D(64, 3, activation='elu'))
model.add(Dropout(keep_prob))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()

# train model
learning_rate = 1.0e-4
batch_size = 50
steps_per_epoch = 500
nb_epoch = 6

model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))

model.fit_generator(batch_generator(data_dir, X_train, y_train, batch_size, True),
                    steps_per_epoch,
                    nb_epoch,
                    max_queue_size=1,
                    validation_data=batch_generator(data_dir, X_valid, y_valid, batch_size, False),
                    validation_steps=len(X_valid),
                    verbose=1) 
print('saving')
model.save('model.h5')