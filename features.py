#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import preprocessing
import python_speech_features

def extract_features(audio, rate):
    # Compute MFCC features from an audio signal.
    mfcc_feat = python_speech_features.mfcc(audio,
                                            samplerate=rate, 
                                            winlen=0.025, winstep=0.01,
                                            numcep=20, appendEnergy=True)

    # Standardization, or mean removal and variance scaling
    mfcc_feat = preprocessing.scale(mfcc_feat)

    # Compute delta features from a feature vector sequence.
    mfcc_delta_feat = python_speech_features.delta(mfcc_feat, 2)

    # Compute delta delta features from a feature vector sequence.
    mfcc_delta_delta_feat = python_speech_features.delta(mfcc_delta_feat, 2)

    return np.hstack((mfcc_feat,
                      mfcc_delta_feat,
                      mfcc_delta_delta_feat))
