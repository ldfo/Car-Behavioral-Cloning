import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
import argparse
import cv2, os
import numpy as np
import matplotlib.image as mpimg
import os

np.random.seed(0)
# seguro es el tamaño de esto:
IM_HEIGHT, IM_WIDTH, IM_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IM_HEIGHT, IM_WIDTH, IM_CHANNELS)

def preprocess(image):
    # remove the sky
    image = image[60:-25, :, :] 
    # resize image
    image = cv2.resize(image, (IM_WIDTH, IM_HEIGHT), cv2.INTER_AREA)
    # rgb2yuv this is what the nvvidia model does
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image

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
                choice = np.random.choice(3)
                if choice == 0:
                    image = mpimg.imread(os.path.join(data_dir, left.strip()))
                    steering_angle += 0.2
                elif choice == 1:
                    image = mpimg.imread(os.path.join(data_dir, right.strip()))
                    steering_angle -= 0.2
                elif choice ==2:
                    image = mpimg.imread(os.path.join(data_dir, center.strip()))
                if np.random.rand() < 0.5:
                    image = cv2.flip(image, 1)
                    steering_angle = -steering_angle

            else:
                # chooses image from center
                image = mpimg.imread(os.path.join(data_dir, center.strip()))
            # add image and steering angle
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers


def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    """
    Load training data and split it into training and validation set
    """
    data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv'))
    data_df.columns = ['center','left','right','steering','throttle','break','speed']
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()


    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                    args.samples_per_epoch,
                    args.nb_epoch,
                    max_queue_size=1,
                    validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                    validation_steps=5,
                    callbacks=[checkpoint],
                    verbose=1)


if __name__ == '__main__':
    main()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE
import argparse
import cv2, os
import numpy as np
import matplotlib.image as mpimg
import os

np.random.seed(0)

IM_HEIGHT, IM_WIDTH, IM_CHANNELS = 66, 200, 3

def preprocess(image):
    # remove the sky
    image = image[60:-25, :, :] 
    # resize image
    image = cv2.resize(image, (IM_WIDTH, IM_HEIGHT), cv2.INTER_AREA)
    # rgb2yuv this is what the nvvidia model does
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image

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
                choice = np.random.choice(3)
                if choice == 0:
                    image = mpimg.imread(os.path.join(data_dir, left.strip()))
                    steering_angle += 0.2
                elif choice == 1:
                    image = mpimg.imread(os.path.join(data_dir, right.strip()))
                    steering_angle -= 0.2
                elif choice ==2:
                    image = mpimg.imread(os.path.join(data_dir, center.strip()))
                if np.random.rand() < 0.5:
                    image = cv2.flip(image, 1)
                    steering_angle = -steering_angle

            else:
                # chooses image from center
                image = mpimg.imread(os.path.join(data_dir, center.strip()))
            # add image and steering angle
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers


def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    """
    Load training data and split it into training and validation set
    """
    data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv'))
    data_df.columns = ['center','left','right','steering','throttle','break','speed']
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()


    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                    args.samples_per_epoch,
                    args.nb_epoch,
                    max_queue_size=1,
                    validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                    validation_steps=5,
                    callbacks=[checkpoint],
                    verbose=1)


if __name__ == '__main__':
    main()
