
# **Behavioral Cloning** 

## Nick Xydes

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./training_history.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Solution
### Model Architecture

#### 1. Model

I tested with variations of the lenet architecture and variations of the NVIDIA architecture. Both of these architectures were discusses in the lectures and I saw in the forums that others suggested using just the NVIDIA architecture so I focused on that.

I ended up only seriously considering the NVIDIA architecture as it was the most highly recommended and honestly the newest. Also, given its complexity increase over the lenet architecture I expected it to perform better. The first portion of the architecture is similar to pre-processing the images. I first normalize the images to a range of -1 to 1 and then crop the top and bottom off of the image. The first step helps the model train more efficiently and the second step gets rid of superflous data that the network shouldn't use when learning. The final architecture looked like so:

- Normalization - Lambda layer
- Cropping - remove the top 60 and bottom 23 pixels
- Convolution - Depth 24, kernel 5x5, stride 2
- RELU Activation
- Convolution - Depth 36, kernel 5x5, stride 2
- RELU Activation
- Convolution - Depth 48, kernel 5x5, stride 2
- RELU Activation
- Convolution - Depth 64, kernel 3x3
- RELU Activation
- Convolution - Depth 64, kernel 3x3
- RELU Activation
- Flatten
- Fully Connected - size 100
- Dropout - 50% keep
- Fully Connected - size 50
- Dropout - 50% keep
- Fully Connected - size 10
- Fully Connected - size 1

The only difference between my architecture and the NVIDIA architecture are adding the dropout layers after the first two Fully Connected layers. These dropout layers helped to prevent overfitting my dataset. I also experimented with adding more fully connected layers, thinking more layers with dropout would help. It ended up causing worse performance. I do think adding more convolutional layers may be beneficial and may experiment with more in the future.

#### 2. Training

I used an adam optimizer to increase the performance and prevent manual training rate tuning. I also ended up using all of the data for each epoch as it didn't take an inconsiderate amount of time to train. I did chop up the training data and took 20% of the data to be validation data, as demonstrated in the examples. My final parameters were as such:

- Epochs: 5
- Batch size: 32
- Examples per epoch: 100% of training set

And my final performance was as such:

![alt text][image1]


### DATA SET 
#### 1. Data Collection

Data was collected using the Udacity training simulator. I generally followed the strategy given in the examples and for the final model did the following:
- 2 laps around counter-clockwise
- 2 laps around clockwise
- twice across the bridge both directions
- twice through the curvy section both directions
- ~ 10 examples of being offset by 30-70 degrees and rapidly correcting back to the center of the lane

At first, I started by driving around once in the counter-clockwise direction and seeing how it went. I slowly started adding more training data to my training set. I found more laps and in both directions did help tremendously. At this point I also started playing with the data augmentation as described in the next chapter and tuning those parameters.

Surprisingly, my models started producing terrible results. I first struggled with the first turn, but a couple mroe laps seemed to help with this problem. I then struggled with the bridge as the car would get confused as soon as it entered. I solved this by driving across the bridge a couple of times and found that it then could get across the bridge quite reliably. Finally, it was struggling with the curvy sections after the bridge so I took more data through there. It was at this point that the model starting performing worse and worse. As I took more data it kepts getting worse. Every time it reached the curvy section after the bridge it would start to drift into the dirt.

So, I eventually got frustrated and said it must have been because of my training data, so I spent about 15-20 minutes and started over. I did the same type of driving, following the same steps as listed above, when I had finished my training set was around 20% larger. I then trained on this data and got an excellent performance first time. I really learned a lot from this specific experience. It really drove home the point to me that this model could only ever be as good as the training data I give it. Garbage in, garbage out.

#### 2. Data Processing

To process the data, I really didn't do too much. I know from the previous projects that pre-processing the data will give the best results, but I decided to play it simpler this time and see how it would turn out. I followed the suggested path pretty closely.

The only image manipulation I did to each image was to crop the top and bottom out. This makes the most sense because it gets rid of the superfluous information, at least for the first track. But to get the best performance of the system I wanted to have the best data. Since most of my data was from the middle of the track I decide to use all of the images, center left and right. 

First, I took each center image and added it with its assigned angle to the image list. Next, I flipped the center image and added the flipped image with the negative of the assigned angle to the image list. Finally, I took the left and right images and added/subtracted a correction value of 0.3 to the assigned angle. 

The flipped image will help generalize the model to both left and right turns. The left and right images will help train the car to stay in the middle of the lane.

