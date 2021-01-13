#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 23:47:07 2021

@author: noamenasheof
"""



import librosa
#from scipy.io import wavfile
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


class LoadData:
    
    def __init__(self, loc1, loc2):
        self.loc1 = loc1
        self.loc2 = loc2
        
        self.data = []
        self.sampling_rate = []
        self.labels = []
        
        for dirname, dirnames, filenames in os.walk(self.loc1):
            for subdirname in filenames:
                cmap = plt.get_cmap('inferno')

                try:
                    subject_path = os.path.join(dirname, subdirname)
                    y, sr = librosa.load(subject_path, mono=True, duration=5)
                    #print(subject_path[-24:-4])
                    plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
                    plt.axis('off');
                    plt.savefig(subject_path[-24:-4]+'.png')
                    plt.clf()

                    self.labels.append(subject_path[-24:])
                except :
                    print(filenames)
                    
        for dirname, dirnames, filenames in os.walk(self.loc2):
            for subdirname in filenames:
                cmap = plt.get_cmap('inferno')

                try:
                    subject_path = os.path.join(dirname, subdirname)
                    y, sr = librosa.load(subject_path, mono=True, duration=5)
                    plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
                    plt.axis('off');
                    plt.savefig(subject_path[-24:-4]+'.png')
                    plt.clf()
                    
                
                    self.labels.append(subject_path[-24:])
                except :
                    print(filenames)
                    
                
if __name__  ==  '__main__' :
    
    
    loc1 = '/Users/noamenasheof/Desktop/Project/Audio_Song_Actors_01-24'
    loc2 = '/Users/noamenasheof/Desktop/Project/Audio_Speech_Actors_01-24'

    files = LoadData(loc1, loc2)                            
    location = '/Users/noamenasheof/Desktop/Project'
    labes_name = 'im_lables.csv'
    lables_data_frame = files.lables_data_frame
    lables_data_frame.to_csv(location + '/' + labes_name)
    

import librosa
#from scipy.io import wavfile
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf    
    


im = '/Users/noamenasheof/Desktop/Project/03-01-01-01-01-01-01.png'
img = tf.keras.preprocessing.image.load_img(im, color_mode='rgb', target_size= (224,224))
img=np.array(img)

im1 = '/Users/noamenasheof/Desktop/Project/03-01-01-01-01-01-02.png'
img1 = tf.keras.preprocessing.image.load_img(im1, color_mode='rgb', target_size= (224,224))
img1=np.array(img1)

    
    
    
    
    