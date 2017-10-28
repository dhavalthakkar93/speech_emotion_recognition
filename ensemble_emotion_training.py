"""ensemble_emotion_training.py: purpose of this script is to train ensemble model based on  Toronto emotion speech dataset to predict the emotion from speech"""

__author__ = "Dhaval Thakkar"
__purpose__ = "Part of Omnisys Solution internship selection process"


import glob
import librosa
import librosa.display
import numpy as np
import _pickle as pickle
from sklearn import svm
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
import pandas as pd


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


def parse_audio_files(path):
    features, labels = np.empty((0, 193)), np.empty(0)
    labels = []
    for fn in glob.glob(path):
        try:
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
        except Exception as e:
            print("Error encountered while parsing file: ", fn)
            continue
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        features = np.vstack([features, ext_features])
        labels = np.append(labels, fn.split('/')[2].split('-')[1].split('.')[0])
    return np.array(features), np.array(labels, dtype=np.int)


tr_features, tr_labels = parse_audio_files('./train_sounds/*.wav')

tr_features = np.array(tr_features, dtype=pd.Series)
tr_labels = np.array(tr_labels, dtype=pd.Series)

model1 = svm.SVC(kernel='linear', C=1000, gamma='auto')
model1.fit(X=tr_features.astype(int), y=tr_labels.astype(int))

model2 = svm.SVC(kernel='rbf', C=1000, gamma='auto')
model2.fit(X=tr_features.astype(int), y=tr_labels.astype(int))

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
estimators = [('linear', model1), ('rbf', model2)]
ensemble = VotingClassifier(estimators)

results = model_selection.cross_val_score(ensemble, tr_features.astype(int), tr_labels.astype(int), cv=kfold)
ensemble.fit(X=tr_features.astype(int), y=tr_labels.astype(int))

filename = 'Ensemble_Model_protocol2.sav'

pickle.dump(ensemble, open(filename, 'wb'), protocol=2)

print('Model Saved..')
