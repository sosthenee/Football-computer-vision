# Football Computer vision & Deep learning
Tracking players and ball on matchs clips, using Tensorflow Object detection API and OpenCV. 
Thanks to computer vison the porject was to detecting the differents Teams playing and their pace.
Then i tried to recognize the ball and fit it's trajectory.

This video inspired me: https://www.youtube.com/watch?v=GrAdG9r7shU&list=WL&index=42 

# Demo
See example gif below of the game of Arsenal against Leicester.
we can identify all the players + referees, the soccer ball and also predict which team the player is based on the color of their jersey. (These teams have obviously a great jersey for recognition on a green pitch)

![](football_deep.gif)


# Some informations 
The API provides pre-trained object detection models that have been trained on the COCO dataset.
I have chosen the SSDLite mobilenet v2 because i was interested in real time analysis. 

I have used the pre-trained model and it's weights but it would be more efficient with additional data for training the model on the defferents players 

The boundarires.py file finds the HSV boundaries manualiy for Team detection

The load_model_data.py file Load pipeline config and build a detection model and restore checkpoint so i can change the model easily


<h2>Developpement tools</h2>
<ul>
<li>Tensorflow 2.0</li>

<li>Python 3.8.6</li>
</ul>


# Important links:

-For openCV circle detection : https://github.com/opencv/opencv/blob/3.4/samples/python/tutorial_code/ShapeDescriptors/bounding_rects_circles/generalContours_demo1.py

https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.html

-For team detection :https://github.com/priya-dwivedi/Deep-Learning/blob/master/soccer_team_prediction/soccer_realtime.ipynb

-Tensorflow for macOS Big Sur : https://github.com/apple/tensorflow_macos

-Tensorflow object detection documentation : https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/

# Future Work
We can imagine getting the pace of the players and of the ball. This could be simpler with a stable footage

We can create a mini-map based on the players positions. 

An idea would be to help blind people to folow the game --> https://www.youtube.com/watch?v=0KE1lIeVBH8
