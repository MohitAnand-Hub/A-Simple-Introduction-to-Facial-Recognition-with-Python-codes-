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

Every Machine Learning algorithm takes a dataset as input and learns from this data. The algorithm goes through the data and identifies patterns in the data.  

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
23.1	               15.8	              (255, 224, 189)             	5.2              	4.4
 
So, our image is now a vector that could be represented as (23.1, 15.8, 255, 224, 189, 5.2, 4.4). Of course there could be countless other features that could be derived from the image (for instance, hair color, facial hair, spectacles, etc). However, for the example, let us consider just these 5 simple features.

Now, once we have encoded each image into a feature vector, the problem becomes much simpler. Clearly, when we have 2 faces (images) that represent the same person, the feature vectors derived will be quite similar. Put it the other way, the “distance” between the 2 feature vectors will be quite small.

Machine Learning can help us here with 2 things:

1. Deriving the feature vector: it is difficult to manually list down all of the features because there are just so many. A Machine Learning algorithm can intelligently label out many of such features. For instance, a complex features could be: ratio of height of nose and width of forehead. Now it will be quite difficult for a human to list down all such “second order” features.
2. Matching algorithms: Once the feature vectors have been obtained, a Machine Learning algorithm needs to match a new image with the set of feature vectors present in the corpus.
Now that we have a basic understanding of how Face Recognition works, let us build our own Face Recognition algorithm using some of the well-known Python libraries.

Case Study
We are given a bunch of faces – possibly of celebrities like Mark Zuckerberg, Warren Buffett, Bill Gates, Shah Rukh Khan, etc. Call this bunch of faces as our “corpus”. Now, we are given image of yet another celebrity (“new celebrity”). The task is simple – identify if this “new celebrity” is among those present in the “corpus”.

As you can see, we have celebrities like Barack Obama, Bill Gates, Jeff Bezos, Mark Zuckerberg, Ray Dalio and Shah Rukh Khan.

It is obvious that this is Shah Rukh Khan. However, for a computer this is a challenging task. The challenge is because of the fact that for us humans, it is easy to combine so many features of the images to see which one is which celebrity. However, for a computer, it isn’t straightforward to learn how to recognize these faces.

There is an amazingly simple Python library that encapsulates all of what we learn above – creating feature vectors out of faces and knowing how to differentiate across faces. This Python library is called as face_recognition and deep within, it employs dlib – a modern C++ toolkit that contains several machine learning algorithms that help in writing sophisticated C++ based applications.

face_recognition library in Python can perform a large number of tasks:

% Find all the faces in a given image
% Find and manipulate facial features in an image
% Identify faces in images
% Real-time face recognition

Here, we will talk about the 3rd use case – identify faces in images.

Implementation in Python

This section contains the code for a building a straightforward face recognition system using the face_recognition library. This is the implementation part, we will go through the code to understand it in more detail in the next section.

# import the libraries
import os
import face_recognition

# make a list of all the available images
images = os.listdir('images')

# load your image
image_to_be_matched = face_recognition.load_image_file('my_image.jpg')

# encoded the loaded image into a feature vector
image_to_be_matched_encoded = face_recognition.face_encodings(
    image_to_be_matched)[0]

# iterate over each image
for image in images:
    # load the image
    current_image = face_recognition.load_image_file("images/" + image)
    # encode the loaded image into a feature vector
    current_image_encoded = face_recognition.face_encodings(current_image)[0]
    # match your image with the image and check if it matches
    result = face_recognition.compare_faces(
        [image_to_be_matched_encoded], current_image_encoded)
    # check if it was a match
    if result[0] == True:
        print "Matched: " + image
    else:
        print "Not matched: " + image
The folder structure is as follows:

facialrecognition:

fr.py
my_image.jpg
images/
barack_obama.jpg
bill_gates.jpg
jeff_bezos.jpg
mark_zuckerberg.jpg
ray_dalio.jpg
shah_rukh_khan.jpg
warren_buffett.jpg
Our root directory, facialrecognition contains:

Our face recognition code above in the form of fr.py.
my_image.jpg – the image to be recognized (“new celebrity”).
images/ – the “corpus”.
When you create the folder structure as above and run the above code, here is what you get as the output:

Matched: shah_rukh_khan.jpg
Not matched: warren_buffett.jpg
Not matched: barack_obama.jpg
Not matched: ray_dalio.jpg
Not matched: bill_gates.jpg
Not matched: jeff_bezos.jpg
Not matched: mark_zuckerberg.jpg
Clearly, the “new celebrity” is Shah Rukh Khan and our face recognition system is able to detect it!

 

Understanding the Python code
Now, let us go through the code to understand how it works:

# import the libraries
import os
import face_recognition
These are simply the imports. We will be using the built-in os library to read all the images in our corpus and we will use face_recognition for the purpose of writing the algorithm.

# make a list of all the available images
images = os.listdir('images')
This simple code helps us identify the path of all of the images in the corpus. Once this line is executed, we will have:

images = ['shah_rukh_khan.jpg', 'warren_buffett.jpg', 'barack_obama.jpg', 'ray_dalio.jpg', 'bill_gates.jpg', 'jeff_bezos.jpg', 'mark_zuckerberg.jpg']
Now, the code below loads the new celebrity’s image:

# load your image
image_to_be_matched = face_recognition.load_image_file('my_image.jpg')
To make sure that the algorithms are able to interpret the image, we convert the image to a feature vector:

# encoded the loaded image into a feature vector

image_to_be_matched_encoded = face_recognition.face_encodings(

    image_to_be_matched)[0]
The rest of the code now is fairly easy:

# iterate over each image
for image in images:
    # load the image
    current_image = face_recognition.load_image_file("images/" + image)

    # encode the loaded image into a feature vector
    current_image_encoded = face_recognition.face_encodings(current_image)[0]

    # match your image with the image and check if it matches
    result = face_recognition.compare_faces(
        [image_to_be_matched_encoded], current_image_encoded)

    # check if it was a match
    if result[0] == True:
        print "Matched: " + image
    else:
        print "Not matched: " + image
Here, we are:

Looping over each image.
Encoding the image into a feature vector.
Comparing the loaded image with the image to be recognized.
If it is a match, we print that. If it is a mismatch, we print that as well.
The output as shown above clearly suggests that this simple face recognition algorithm works amazingly well. Let us try replacing my_image with another image:



When you run the algorithm again, you will see the following output:

Not matched: shah_rukh_khan.jpg
Not matched: warren_buffett.jpg
Not matched: barack_obama.jpg
Not matched: ray_dalio.jpg
Not matched: bill_gates.jpg
Not matched: jeff_bezos.jpg
Not matched: mark_zuckerberg.jpg
Clearly, the system did not identify Jack Ma as any of the above celebrities. This indicates that our algorithm is quite good in both:

Correctly identifying those that are present in the corpus
Flagging a mismatch for those that are not present in the corpus
 

Face Recognition Applications
Face Recognition is a well researched problem and is widely used in both industry and in academia. As an example, a criminal in China was caught because a Face Recognition system in a mall detected his face and raised an alarm. Clearly, Face Recognition can be used to mitigate crime. There are many other interesting use cases of Face Recognition:

Facial Authentication: Apple has brought in Face ID for Facial Authentication in iPhones. Some of the leading banks are trying to use Facial Authentication for lockers.
Customer Service: Some of the banks in Malaysia have installed systems which use Face Recognition to detect valuable customers of the bank so that the bank can provide the personalized service. This way, banks are able to generate more revenues by retaining such customers and keeping them happy.
Insurance Underwriting: Many insurance companies are using Face Recognition to match the face of the person with that provided in the photo ID proof. This way, the underwriting process becomes much faster.
 

End Notes

To summarize, Face Recognition is an interesting problem with lots of powerful use cases which can significantly help society across various dimensions. While there will always be an ethical risk attached to commercialzing such techniques, that is a debate we will shelve for another time.
