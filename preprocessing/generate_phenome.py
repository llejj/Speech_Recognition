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
output_dir = '../phoneme_data_wav'
train_output_dir = os.path.join(output_dir, 'TRAIN')
test_output_dir = os.path.join(output_dir, 'TEST')

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

def trim(audio_path, target_path, start_sample, end_sample):
    length = int(end_sample) - int(start_sample)
    command = f"sox {audio_path} {target_path} trim {start_sample}s {length}s"
    subprocess.call(command, shell=True)

def remove(audio_path):
    command = f"rm {audio_path}"
    subprocess.call(command, shell=True)




def extract_phoneme_data(audio_path, target_dir, index):
    transcript_path = os.path.splitext(audio_path)[0] + '.PHN'
    file = open(transcript_path, 'r')

    i = 0
    for line in file:
        words = line.split()  # Split the line into individual words
        last_word = words[-1]  # Get the phoneme
        phoneme_dir = os.path.join(target_dir, last_word) # Get the phoneme_directory
        os.makedirs(phoneme_dir, exist_ok=True) # Make the directory if it doesn't exist

        # Extract trimmed wav file
        convert_nist_to_wav(audio_path, os.path.join(phoneme_dir, 'temp.WAV'))
        trim(os.path.join(phoneme_dir, 'temp.WAV'), os.path.join(phoneme_dir, str(index) + "_" + str(i) + ".WAV"), words[0], words[1])
        remove(os.path.join(phoneme_dir, 'temp.WAV'))

        # convert to mfcc
        """
        mfcc_features = extract_mfcc(os.path.join(phoneme_dir, str(index) + "_" + str(i) + ".WAV"))
        remove(os.path.join(phoneme_dir, str(index) + "_" + str(i) + ".WAV"))
        np.save(os.path.join(phoneme_dir, str(index) + "_" + str(i)), mfcc_features)
        """

        i += 1

    file.close()

# Preprocess training data
i=0
for subdir, _, files in os.walk(train_dir):
    for file in files:
        if os.path.splitext(file)[1] == '.WAV':
            audio_path = os.path.join(subdir, file)
            extract_phoneme_data(audio_path, train_output_dir, i)
            print(i)
            i += 1

# Preprocess testing data
i=0
for subdir, _, files in os.walk(test_dir):
    for file in files:
        if os.path.splitext(file)[1] == '.WAV':
            audio_path = os.path.join(subdir, file)
            extract_phoneme_data(audio_path, test_output_dir,i)
            print(i)
            i += 1