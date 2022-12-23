# APL
 Projects from interning @ APL under Dr. Amit Banerjee (Jan '22-Aug '22)
 
 **Project I: Random Forest Hyperspectral Imaging Classifier**
 
In this project, I used Python and Keras to build a high-accuracy random forest classifier that identified whether hyperspectral images of stool samples could provide enough evidence to detect blood in the stool, and early sign of colon cancer. The goal was to use the random forest to allow doctors and consumers to quickly and noninvasively find these early signatures, without having to run expensive, time-intensive tests for every result.

The classifier looked at the hundred strongest spectral signatures/markers of blood, and determined based on their relative intensities whether there was a statistically significant amount of blood. Even when trained on very little training data (n=10 samples), the classifier achieved 96% accuracy when compared to real tests run by Johns Hopkins to determine the presence of blood. Dr. Banerjee is currently seeking funding to further develop the project for consumer use, primarily in app development and acquiring low-cost hyperspectral lenses for smartphones to allow patients to use the app in the privacy of their homes.
 
 **Project II: Transformer-CNN Video Classifier**

I then continued working with Dr. Banerjee to develop a video-based action-classifying neural network, with the end goal of achieving realtime action classification in mice brain-computer interfaces. The hypothesis behind the research was that by looking at video feeds of a mouse's brain, a program could detect subtle changes in the brain's vascular circulation, and eventually learn which circulation patterns correspond to different actions. In order to achieve this extremely fine-grained level of video classification, I implemented a state-of-the-art CNN-Transformer architecture in Tensorflow.

I first began by implementing RCNN, CNN, and Transformer models using Pytorch and Tensorflow, and trained them on human action datasets in order to establish a solid foundation in action classification. After lots of trial, error, and iteration, my final model was able to achieve an astounding 95% classification accuracy on the UCF101 human action dataset. Higher accuracy could have certainly been achieved via pose estimation and LSTM approaches, however these would not be useful for the mide brain-computer interfaces. Currently, my CNN-Transformer model is being optimized for and applied to the mice brain-computer interfaces.
