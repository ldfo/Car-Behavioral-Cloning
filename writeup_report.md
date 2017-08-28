# **Behavioral Cloning**

## Writeup

---

The goals of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./nvidia_cnn.png "Nvidia's diagram of their network"
[image2]: ./center_driving.jpg "Grayscaling"
[image3]: ./recovery_1.jpg "Recovery Image"
[image4]: ./recovery_2.jpg "Recovery Image"
[image5]: ./recovery_3.jpg "Recovery Image"
[image6]: ./steering_distribution.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model I chose was based on nvidia's network architecture described on [this article](https://arxiv.org/pdf/1604.07316v1.pdf "End-to-End Deep Learning for Self-Driving Cars")

The model consists of a convolution neural network composed by a cropping layer followed by a normalization layer, 5 convolutional layers and 4 fully connected layers (model.py lines 76-90)

The model includes ELU layers to introduce nonlinearity, I chose ELU instead of RELU because I saw marginally faster convergence rates. The data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, with a learning rate of 1.0e-4 (line 98 of model.py).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... Training data was chosen to keep the vehicle driving on the road. First I drove on the center of the track on one direction and then on the opposite direction. I also recorded the car recovering from the left and right sides of the track.
Then I noticed my training data was predominantly straight driving (steering = 0), so I got rid of all the data where steering = 0 just for balancing the dataset and for the algorithm to learn more easily to make turns. This helped the car to steer correctly.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I read nvidia's article so I decided that was a good starting point for the network.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The first model I trained didn't perform well at all, the car veered off the track immediatly and I realized that was because I was cropping outside the network with opencv and I wasn't cropping on the drive.py so I decided to include all the preprocessing of the network inside keras.

Then I had to tune the hyperparameters for getting good convergence and with decent hyperparameters and the dropout layer I was getting a decently low mean squared error.

But when I tested the car on the track it was just driving straight and falling into the lake. I tried augmenting the training data by randomly choosing from left right and center images and correcting for the steering with +- .02 (see function augment_images on line 17 in model.py). Also I implemented a random flip on the image.
This improvements helped the car to get past the first curves but it kept veering of the track.

I realized that my training data was 80% steering = 0 so I got rid of all the data where steering = 0 and it helped a lot.

The car drove almost all the track but fell off on certain spots so I recorded more recovery data from the parts where the car was falling off and I got the car driving all the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 76-90) consisted of a CNN based on nvidia's network. Here is an image from nvidia's article showing the network's architecture.

```
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
```

![Nvidia's diagram of their CNN][image1]

#### 3. Creation of the Training Set & Training Process

I first recorded one lap on track one using center lane driving and one lap driving the other way. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive from the sides to the center in case it got too close to the edge of the track. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also applied a random flip to the training images and angles thinking that this would capture the behavior of the car turning both sides better.

After the collection process, I had 3487 data points. I observed the distribution of the steering and decided to drop the data points with steering = 0.
Here is the distribution before and after this procedure:
![steering distribution][image6]

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
The hyperparameters I used were:
```learning_rate = 1.0e-4
batch_size = 50
steps_per_epoch = 500
nb_epoch = 6
```
