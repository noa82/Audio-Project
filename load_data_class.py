#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 22:40:02 2020

@author: noamenasheof
"""
import numpy as np
import pandas as pd
from scipy.io import wavfile
import os
import python_speech_features 
#from tqdm import tqdm





class Data:
    def __init__(self,location1,location2):
        
        self.location1 = location1
        self.location2 = location2
        
        self.max_len = 0
        self.files = []
        self.check = []
        self.labels = []
        self.dontwork = []
        self.emotions = {'01' : 'neutral', '02' : 'calm', '03' : 'happy', '04' : 'sad',
            '05' : 'angry', '06' : 'fearful', '07' : 'disgust', '08' : 'surprised'}
        for dirname, dirnames, filenames in os.walk(self.location1):
            for subdirname in filenames:
                subject_path = os.path.join(dirname, subdirname)
                wave = self.load_wave(subject_path) 
                #temp = []
                #down sample the audio
                if wave != -1:
                    frame_rate = wave[0]
                    data = wave[1]
                    if data[0].shape!=():
                        data=data[:,0]
                    
                    n=len(data)
                    
                    data = data/(32767+32768)
                    mask = self.envelope(data,frame_rate,0.001)
                    data=data[mask]
                    
                    
                    # freq = np.fft.rfftfreq(n,1/rate)
                    # Y = abs(np.fft.rff(data)/n)
                    
                    self.files.append(data)
                    self.check.append(frame_rate)
                    self.labels.append(subject_path[-24:])
                else:
                    print(subject_path)
                    self.dontwork.append(subject_path)
                    

        for dirname, dirnames, filenames in os.walk(self.location2):
            for subdirname in filenames:
                subject_path = os.path.join(dirname, subdirname)
                wave = self.load_wave(subject_path) 
                if wave != -1:
                    frame_rate = wave[0]
                    data = wave[1]
                    if data[0].shape!=():
                        data=data[:,0]
                    
                
                    n=len(data)
                    
                    #####################################################################
                    data = data/(32767+32768)
                    mask = self.envelope(data,frame_rate,0.0001)
                    data=data[mask]
                    
                    
                    # freq = np.fft.rfftfreq(n,1/rate)
                    # Y = abs(np.fft.rff(data)/n)
                    
                    self.files.append(data)
                    self.check.append(wave[0])
                    self.labels.append(subject_path[-24:])
                else:
                    print(subject_path)
                    self.dontwork.append(subject_path)
                    

        self.frame_rate = self.check[0]
                    
        for file in self.files:
            if len(file) > self.max_len:
                self.max_len = len(file)
                
                # make them all the same length        
        for i in range(len(self.files)):
            if len(self.files[i]) < self.max_len:
                diff = self.max_len - len(self.files[i])
                zeros = np.zeros(diff)
                self.files[i] = np.concatenate((self.files[i], zeros))
        
        self.files = np.array(self.files)  
        
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
            
        self.MFCC_data=[]
        self.logfbank_data = []
        self.delta_data = [] 

        

            
    def envelope(self,y,rate,threshold):
        mask = []
        y=pd.Series(y).apply(np.abs)
        y_mean = y.rolling(window = int(rate/10),min_periods=1,center= True).mean()
        for mean in y_mean:
            if mean>threshold:
                mask.append(True)
            else:
                mask.append(False)
        return mask
        
        
        
    
    def load_wave(self,wave_filename):
        try :
            frame_rate, data = wavfile.read(wave_filename)
            # if data.dtype == np.uint8 :
            #     data = (data.astype(np.int16)-128)*256
            # elif data.dtype != np.int16 : 
            #     raise Exception('Unhandeled sample width')
            return frame_rate, \
                   data
        except KeyboardInterrupt:
            raise
        except :
            return -1
    
    def to_data_frame(self):
        files1 = pd.DataFrame(self.files)
        lables_data_frame = pd.DataFrame({'Vocal_channel': self.vocal_channel,
                                  'emotional_intensity': self.emotional_intensity,
                                  'statement': self.statement,'Repetition': self.repetition,
                                  'gender': self.gender,'actor_number':self.actor_number,
                                  'emotion': self.emotion})
        return files1 , lables_data_frame
        
    def to_MFCC(self,data):
        # Compute MFCC features from an audio signal.
        for i in range(len(self.files)):
            temp = python_speech_features.mfcc(data.iloc[i,:], self.frame_rate, nfilt = 50 )
            self.MFCC_data.append(np.array(temp[1:-1]))
          
        
        self.MFCC_data = np.array(self.MFCC_data)
        return self.MFCC_data
    
    def to_logfbank(self,data):
        # Compute logfbank features from an audio signal.
        for i in range(len(self.files)):
            temp = python_speech_features.logfbank(data.iloc[i,:], self.frame_rate, nfilt = 40 , nfft = int(self.frame_rate/40))
            self.logfbank_data.append(np.array(temp[1:-1]))
          
        
        self.logfbank_data = np.array(self.logfbank_data)
        return self.logfbank_data
    
    
    
    def to_delta(self,log_fbank):
        # Compute logfbank features from an audio signal.
        for i in range(len(self.files)):
            temp = python_speech_features.delta(log_fbank[i], 2)
            self.delta_data.append(np.array(temp[1:-1]))
          
        
        self.delta_data = np.array(self.delta_data)
        return self.delta_data
    
    
        
        
    def save_as_dataframe(self,location,files_name,labes_name,data):
        files1 = pd.DataFrame(data)
        lables_data_frame = pd.DataFrame({'Vocal_channel': self.vocal_channel,
                                  'emotional_intensity': self.emotional_intensity,
                                  'statement': self.statement,'Repetition': self.repetition,
                                  'gender': self.gender,'actor_number':self.actor_number,
                                  'emotion': self.emotion})
        lables_data_frame.to_csv(location + '/' + labes_name)

        files1.to_csv(location + '/' + files_name)
        return files1
        
if __name__ == '__main__':
    
    path1 = '/Users/noamenasheof/Desktop/Project/Audio_Song_Actors_01-24'
    path2 = '/Users/noamenasheof/Desktop/Project/Audio_Speech_Actors_01-24'
    
    wave_data = Data(path1,path2)
    print('data frame')
    data,lables = wave_data.to_data_frame()
    print('ok')
    MFCC_data = wave_data.to_MFCC(data)
    MFCC_data_temp = []
    for i in range(len(MFCC_data)):
        MFCC_data_temp.append(np.array(MFCC_data[i].flatten()))
    
    MFCC_data_temp = np.array(MFCC_data_temp)
        
    # import matplotlib.pyplot as plt

    
    # y = data.iloc[0,:]
    # plt.plot(y)
    
    
    
    print('Saving as CSV files')
    files1 = wave_data.save_as_dataframe('/Users/noamenasheof/Desktop/Project', 'wave_all_new.csv', 'labels_all_new.csv',MFCC_data_temp)
    print('CSV files saved')


    
    
    
    # log_data = wave_data.to_logfbank(data)
    # log_data_temp = []
    # for i in range(len(log_data)):
    #     log_data_temp.append(np.array(log_data[i].flatten()))
    
    # log_data_temp =np.array(log_data_temp)
        
    # import matplotlib.pyplot as plt

    
    # y = data.iloc[0,:]
    # plt.plot(y)
    
    
    
    # print('Saving as CSV files')
    # files1 = wave_data.save_as_dataframe('/Users/noamenasheof/Desktop/Project', 'wave_data_class_all_log.csv', 'labels_data_class_all_log.csv',log_data_temp)
    # print('CSV files saved')
    
    # log_data = wave_data.to_logfbank(data)
    # log_data_temp = []
    # for i in range(len(log_data)):
    #     log_data_temp.append(np.array(log_data[i].flatten()))
    
    # log_data_temp =np.array(log_data_temp)
    # location = '/Users/noamenasheof/Desktop/Project'
    
    # delta = wave_data.to_delta(log_data)
    
    # delta_temp = []
    # for i in range(len(delta)):
    #      delta_temp.append(np.array(delta[i].flatten()))
    # delta_temp =np.array(delta_temp)
    
    
    
    
    # log_data_temp = pd.DataFrame(log_data_temp)
    
    # log_data1 = pd.DataFrame(log_data_temp.iloc[:,:10000])
    
    # log_data2= pd.DataFrame(log_data_temp.iloc[:,10000:])
   
    # log_data1.to_csv(location + '/' + 'log_data1.csv')
    # log_data12.to_csv(location + '/' + 'log_data2.csv')



    # delta_temp = pd.DataFrame(delta_temp)
    
    # delta1 = pd.DataFrame(delta_temp.iloc[:,:10000])
    
    # delta2= pd.DataFrame(delta_temp.iloc[:,10000:])
   
    # delta1.to_csv(location + '/' + 'delta1.csv')
    # delta2.to_csv(location + '/' + 'delta2.csv')

#wave_data.frame_rate
# print(sum(files1.iloc[0,:] == wave_data.files[0])/len(wave_data.files[0]))
# files = wave_data.files
# files2 = pd.DataFrame(files)

# files1.to_csv('/Users/noamenasheof/Desktop/Project/noa.csv')

# files2 = files1
        
# sum(files1.iloc[0,:]==files[0])/len(files[0])       
        
        
        