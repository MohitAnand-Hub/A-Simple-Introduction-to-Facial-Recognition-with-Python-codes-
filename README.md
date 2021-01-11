# A-Simple-Introduction-to-Facial-Recognition-with-Python-codes-
Introduction
Did you know that every time you upload a photo to Facebook, the platform uses facial recognition algorithms to identify the people in that image? Or that certain governments around the world use face recognition technology to identify and catch criminals? I don’t need to tell you that you can now unlock smartphones with your face!

The applications of this sub-domain of computer vision are vast and businesses around the world are already reaping the benefits. The usage of face recognition models is only going to increase in the next few years so why not teach yourself how to build one from scratch?



In this article, we are going to do just that. We will first understand the inner workings of face recognition, and then take a simple case study and implement it in Python. By the end of the article you will have built your very first facial recognition model!

 

Table of Contents
Understanding how Face Recognition works
Case Study
Implementation in Python
Understanding the Python code
Applications of Facial Recognition Algorithms
 

Understanding how Face Recognition works
In order to understand how Face Recognition works, let us first get an idea of the concept of a feature vector.

Every Machine Learning algorithm takes a dataset as input and learns from this data. The algorithm goes through the data and identifies patterns in the data. For instance, suppose we wish to identify whose face is present in a given image, there are multiple things we can look at as a pattern:

Height/width of the face.
Height and width may not be reliable since the image could be rescaled to a smaller face. However, even after rescaling, what remains unchanged are the ratios – the ratio of height of the face to the width of the face won’t change.
Color of the face.
Width of other parts of the face like lips, nose, etc.
Clearly, there is a pattern here – different faces have different dimensions like the ones above. Similar faces have similar dimensions. The challenging part is to convert a particular face into numbers – Machine Learning algorithms only understand numbers. This numerical representation of a “face” (or an element in the training set) is termed as a feature vector. A feature vector comprises of various numbers in a specific order.

As a simple example, we can map a “face” into a feature vector which can comprise various features like:

Height of face (cm)
Width of face (cm)
Average color of face (R, G, B)
Width of lips (cm)
Height of nose (cm)
Essentially, given an image, we can map out various features and convert it into a feature vector like:

Height of face (cm)	Width of face (cm)	Average color of face (RGB)	Width of lips (cm)	Height of nose (cm)
23.1	15.8	(255, 224, 189)	5.2	4.4
 

So, our image is now a vector that could be represented as (23.1, 15.8, 255, 224, 189, 5.2, 4.4). Of course there could be countless other features that could be derived from the image (for instance, hair color, facial hair, spectacles, etc). However, for the example, let us consider just these 5 simple features.

Now, once we have encoded each image into a feature vector, the problem becomes much simpler. Clearly, when we have 2 faces (images) that represent the same person, the feature vectors derived will be quite similar. Put it the other way, the “distance” between the 2 feature vectors will be quite small.

Machine Learning can help us here with 2 things:

Deriving the feature vector: it is difficult to manually list down all of the features because there are just so many. A Machine Learning algorithm can intelligently label out many of such features. For instance, a complex features could be: ratio of height of nose and width of forehead. Now it will be quite difficult for a human to list down all such “second order” features.
Matching algorithms: Once the feature vectors have been obtained, a Machine Learning algorithm needs to match a new image with the set of feature vectors present in the corpus.
Now that we have a basic understanding of how Face Recognition works, let us build our own Face Recognition algorithm using some of the well-known Python libraries.

 

Case Study
We are given a bunch of faces – possibly of celebrities like Mark Zuckerberg, Warren Buffett, Bill Gates, Shah Rukh Khan, etc. Call this bunch of faces as our “corpus”. Now, we are given image of yet another celebrity (“new celebrity”). The task is simple – identify if this “new celebrity” is among those present in the “corpus”.

Here are some of the images in the corpus:



As you can see, we have celebrities like Barack Obama, Bill Gates, Jeff Bezos, Mark Zuckerberg, Ray Dalio and Shah Rukh Khan.

Now, here is the “new celebrity”:



Note: all of the above images have been taken from Google images.

It is obvious that this is Shah Rukh Khan. However, for a computer this is a challenging task. The challenge is because of the fact that for us humans, it is easy to combine so many features of the images to see which one is which celebrity. However, for a computer, it isn’t straightforward to learn how to recognize these faces.

There is an amazingly simple Python library that encapsulates all of what we learn above – creating feature vectors out of faces and knowing how to differentiate across faces. This Python library is called as face_recognition and deep within, it employs dlib – a modern C++ toolkit that contains several machine learning algorithms that help in writing sophisticated C++ based applications.

face_recognition library in Python can perform a large number of tasks:

Find all the faces in a given image
Find and manipulate facial features in an image
Identify faces in images
Real-time face recognition
Here, we will talk about the 3rd use case – identify faces in images.
