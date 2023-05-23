import os
import numpy as np
from scipy.io import wavfile as wav
from python_speech_features import mfcc
import subprocess

# Define paths to TIMIT dataset directories
timit_dir = '../TIMIT'
train_dir = os.path.join(timit_dir, 'TRAIN')
test_dir = os.path.join(timit_dir, 'TEST')

# Define output directories for processed data
output_dir = '../nist'
train_output_dir = os.path.join(output_dir, 'train')
test_output_dir = os.path.join(output_dir, 'test')

# Create output directories if they don't exist
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

# Define parameters for MFCC extraction
num_cepstral = 13
frame_length = 0.025  # 25 milliseconds
frame_step = 0.01  # 10 milliseconds

# Function to extract MFCC features from an audio file
def extract_mfcc(audio_path):
    (rate, signal) = wav.read(audio_path)
    mfcc_features = mfcc(signal, rate, numcep=num_cepstral, winlen=frame_length, winstep=frame_step)
    return mfcc_features

# Function to convert NIST to WAV format
def convert_nist_to_wav(nist_path, wav_path):
    command = f"sox {nist_path} -t wav {wav_path}"
    subprocess.call(command, shell=True)

# Preprocess training data
"""
for speaker_dir in os.listdir(train_dir):
    speaker_path = os.path.join(train_dir, speaker_dir)
    output_speaker_dir = os.path.join(train_output_dir, speaker_dir)
    os.makedirs(output_speaker_dir, exist_ok=True)

    for subdir, _, files in os.walk(speaker_path):
        for file in files:
            if os.path.splitext(file)[1] == '.WAV':
                nist_path = os.path.join(subdir, file)
                wav_path = os.path.join(output_speaker_dir, file)
                convert_nist_to_wav(nist_path, wav_path)
                """

# Preprocess testing data
for subdir, _, files in os.walk(test_dir):
    for file in files:
        if os.path.splitext(file)[1] == '.WAV':
            nist_path = os.path.join(subdir, file)
            wav_path = os.path.join(test_output_dir, file)
            convert_nist_to_wav(nist_path, wav_path)