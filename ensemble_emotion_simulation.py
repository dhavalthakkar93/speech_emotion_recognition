"""ensemble_emotion_simulation.py: purpose of this script is to predict the emotion from the speech using traind ensemble model based Toronto emotion speech dataset"""

__author__ = "Dhaval Thakkar"
__purpose__ = "Part of Omnisys Solution internship selection process"

import glob
import librosa
import librosa.display
import numpy as np
import _pickle as pickle
import pandas as pd

classes = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


target_files = []


def parse_audio_files(path):
    features = np.empty((0, 193))
    for fn in glob.glob(path):
        try:
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
        except Exception as e:
            print("Error encountered while parsing file: ", fn)
            continue
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        features = np.vstack([features, ext_features])
        target_files.append(fn)
    return np.array(features)


ts_features = parse_audio_files('./new_test_sounds/*.wav')
tr_features = np.array(ts_features, dtype=pd.Series)

filename = 'Ensemble_Model_protocol2.sav'
model = pickle.load(open(filename, 'rb'))

prediction = model.predict(ts_features)

for i, val in enumerate(prediction):
    print("Input File: ", target_files[i], "|", " Predicted Emotion Is:", classes[int(val)])
