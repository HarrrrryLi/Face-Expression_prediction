# Face-Expression_prediction
Developed by Fengru Li, Xiaoyi Tang
## Motivation
Nonverbal behaviour signals, such as facial expressions, provide key information about what we think, act or react. It is an attractive but also challenging to study the signals because they are always hidden or may vary from different people. In this project, we are trying to use machine learning methods for modelling human facial expressions. Therefore, we can get a framework which enables us to predict the facial expression of a never-seen-person when we only hear that person speak.
## Technical details
#### Datasets: 
[RAVDESS](https://smartlaboratory.org/ravdess/)
#### Framework Architecture:
Our project mainly contains three parts: 
- feature extraction for audio and video (Fast Fourier Transform(FFT) and landmarks transformation)
- mapping from speech to facial feature (CNN + RNN)
## Instruction
#### Preprocess Dataset:
- Put all speech files in ./speech folder
- Put train video files in ./train/video folder
- Put test video files in ./test/video folder
- Use python3 to run preprocess_train.py and preprocess_test.py to get feature array, such as: python3 preprocess_train.py   OR  python3 preprocess_test.py
#### Train CNN:
- Create folder called CNN_record in the same path(./)
- Use python3 to run CNN_train.py, such as: python3 CNN_train.py
#### Evaluation CNN:
- Use python3 to run CNN_eval.py, such as: python3 CNN_train.py
#### Make Prediction of landmarks:
- Creat folder called predict_landmark
- Use python3 to run predict.py with arguments which is an speech file endwith .wav , such as: python3 predict.py ./speech/03-01-01-01-01-01-01.wav
- The output is some images and a gif files. Images are in predict_landmark folder. GIF file is in the same path called landmarks.gif
