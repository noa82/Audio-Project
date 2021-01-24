#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 22:44:14 2021

@author: noamenasheof
"""


import tensorflow as tf
import IPython.display as ipd
import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np

path = '/Users/noamenasheof/Downloads/model2 (1).h5'

model = tf.keras.models.load_model(path)



samplerate = 22050  
duration = 4 # seconds
filename = 'quran.wav'
print("Please read one of these sentences: Kids are talking by the door Dogs are sitting by the door. ")
mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,channels=1, blocking=True)
mydata1 = mydata.reshape((88200))
print("end")

#data1, sampling_rate1 = librosa.load(subject_path)
mydata1 = np.mean(librosa.feature.mfcc(y=mydata1, sr=samplerate, n_mfcc=40).T,axis=0) 
mydata1 = mydata1.reshape((1,40,1))

emotions = {0 : 'neutral', 1 : 'calm', 2 : 'happy', 3 : 'sad',4 : 'angry', 5 : 'fearful', 6 : 'disgust', 7 : 'surprised'}

y = model.predict(mydata1)
y_pred_raw = np.argmax(y)
print(emotions[y_pred_raw])

# (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").

model.summary()

