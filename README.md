# OCR-Keras

Implementing Keras-OCR

In order to extract the texts from the images more accurately keras-OCR is implemented on the same set of images tested on Tesseract. Keras-OCR is a Optical Character Recognition tool that comes under Tensorflow. Keras-OCR is open source and is compared to several other OCR including Tesseract and Calamari-OCR. The comparison among them found that the Keras-OCR has best accuracy. 

https://dl.acm.org/doi/abs/10.1145/3475720.3484443

Keras-OCR provides the user with a complete pipeline where user can use either pretrained model or create a new OCR-model. In this documentation I am using pretrained model to recognize the characters in the image.

Prerequisites:

Import keras_ocr
Install opencv-python-headless

The image processing is done as required to get maximum accuracy. Due to the higher demand of CPU usage by Keras-OCR, my computer faced major limitations. These limitations include:

Slow recognition.
Insufficient memory
Unable to perform processing on more than one image [Error: bad memory allocation ]

Keras-OCR uses Deep Neural Network (DNN) which takes major performance hit when performed in low end PC.

2023-01-19 14:23:19.745149: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 733556736 exceeds 10% of free system memory.
2023-01-19 14:23:20.955008: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 733556736 exceeds 10% of free system memory.
2023-01-19 14:23:21.256365: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 733556736 exceeds 10% of free system memory.
2023-01-19 14:23:22.914734: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 733556736 exceeds 10% of free system memory.
2023-01-19 14:23:23.415767: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 366778368 exceeds 10% of free system memory.

Other problems faced during testing are:

Incompatibility of cv2 with opencv-python-headless

Output:
![keras-ocr](https://user-images.githubusercontent.com/99968233/213422070-bba0a1bd-dd3b-4436-8648-e7b931a65bed.png)

Although the region detection is good, actual character recognition falls behind Tesseract often making spelling mistakes. 

![Screenshot from 2023-01-19 14-51-22](https://user-images.githubusercontent.com/99968233/213422089-d219ba89-5c69-4c56-8fbc-239a88dd6de9.png)

Detected Text: 
research
networks
originally
sensor
was
on
mot
vated
applications
by
milntary
exampies
military
of
netwons
sensor
range
ustic
from
argerscale
surveilance
ystems
aco
s
of
surveilance
to
small
tor
ne
worrs
ocean
for
detection
nded
ground
unalte
ground
targert
sensors

![Screenshot from 2023-01-19 16-10-06](https://user-images.githubusercontent.com/99968233/213422096-21910617-f0dc-418c-9b9f-b317a9699eb5.png)
