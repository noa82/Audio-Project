#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 13:33:39 2021

@author: noamenasheof
"""



import librosa
#from scipy.io import wavfile
import numpy as np
import os
import pandas as pd


class LoadData:
    
    def __init__(self, loc1, loc2):
        self.loc1 = loc1
        self.loc2 = loc2
        
        self.data = []
        self.sampling_rate = []
        self.labels = []
        
        for dirname, dirnames, filenames in os.walk(self.loc1):
            for subdirname in filenames:
                
                try:
                    subject_path = os.path.join(dirname, subdirname)
                    data1, sampling_rate1 = librosa.load(subject_path)
                    #mfccs = np.mean(librosa.feature.mfcc(y=data1, sr=sampling_rate1, n_mfcc=40).T,axis=0) 

                    mfccs = librosa.feature.mfcc(y=data1, sr=sampling_rate1, n_mfcc=40).T
                    self.data.append(mfccs)
                    self.sampling_rate.append(sampling_rate1)
                    self.labels.append(subject_path[-24:])
                except :
                    print(filenames)
                    
        for dirname, dirnames, filenames in os.walk(self.loc2):
            for subdirname in filenames:
                try:
                    subject_path = os.path.join(dirname, subdirname)
                    data1, sampling_rate1 = librosa.load(subject_path)
                    mfccs = librosa.feature.mfcc(y=data1, sr=sampling_rate1, n_mfcc=40).T

                    #mfccs = np.mean(librosa.feature.mfcc(y=data1, sr=sampling_rate1, n_mfcc=40).T,axis=0) 
                    self.data.append(mfccs)
                    self.sampling_rate.append(sampling_rate1)
                    self.labels.append(subject_path[-24:])
                except :
                    print(filenames)
                    
        self.data = np.array(self.data)  

        
        self.actor_number = []
        # (01 = speech, 02 = song)
        self.vocal_channel = []
        # (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
        self.emotion = []
        # (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
        self.emotional_intensity = []
        # (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
        self.statement = []
        # (01 = 1st repetition, 02 = 2nd repetition).
        self.repetition = []
        # 0 male 1 female
        self.gender = []

        for lable in self.labels:
            self.vocal_channel.append(lable[3:5])
            self.emotion.append(lable[6:8])
            self.emotional_intensity.append(lable[9:11])
            self.statement.append(lable[12:14])
            self.repetition.append(lable[15:17])
            self.actor_number.append(lable[18:20])
            #(01 to 24. Odd numbered actors are male, even numbered actors are female).
            if int(lable[18:20])%2 == 0:
                self.gender.append(1)
            else: self.gender.append(0)
            
        self.lables_data_frame = pd.DataFrame({'Vocal_channel': self.vocal_channel,
                                  'emotional_intensity': self.emotional_intensity,
                                  'statement': self.statement,'Repetition': self.repetition,
                                  'gender': self.gender,'actor_number':self.actor_number,
                                  'emotion': self.emotion})
                    
  

                
                
if __name__  ==  '__main__' :
    
    
    loc1 = '/Users/noamenasheof/Desktop/Project/Audio_Song_Actors_01-24'
    loc2 = '/Users/noamenasheof/Desktop/Project/Audio_Speech_Actors_01-24'

    files = LoadData(loc1, loc2)        
    print(files.data.shape)
    
    
    location = '/Users/noamenasheof/Desktop/Project'
    labes_name = 'lables_data_frame_new.csv'
    lables_data_frame = files.lables_data_frame
    lables_data_frame.to_csv(location + '/' + labes_name)
    
    data = files.dat
    MFCC_data_temp = []
    for i in range(len(data)):
        MFCC_data_temp.append(np.array(data[i].flatten()))
    
    
    MFCC_data_temp = np.array(data)
    data_name = 'data_frame_new.csv'
    
    MFCC_data_temp.to_csv(location + '/' + data_name)

        

# r = '03-02-03-02-02-02-01.wav'

# data1, sampling_rate1 = librosa.load(loc2 + '/' + r)


                    
                    
                    
                    
                    
                    
                    
                    
                    