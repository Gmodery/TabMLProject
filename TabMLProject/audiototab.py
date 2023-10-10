import os, time, torch, cv2, librosa, librosa.display
import matplotlib.pyplot as plt
import pathlib
import numpy as np
import cv2
from ultralytics import YOLO
from itertools import chain

class_dict = {
    "x": "-",
    "w": "0",
    "r": "1",
    "t": "2",
    "y": "3"
}


# -- Set up model --
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model = YOLO('model\\best.pt')

names = model.names
predicteds = []

# -- Load audio --
audio_file = "audio\\Pentatonic.wav"

n_fft = 2048
hop_length = 32
f_min = 60
f_max = 600

y, sr = librosa.load(audio_file)

y, index = librosa.effects.trim(y)

# -- Get onsets --
onset_times = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, units="time", delta=0.25, wait=150)

duration = librosa.get_duration(y=y, sr=sr)
onset_times = np.append(onset_times, round(duration, 1)-0.1)

c = 0
# -- Loop for every segment between onsets --
for index in range(len(onset_times)-1):
    # -- Trim segments --
    start_sample = librosa.time_to_samples(onset_times[index], sr=sr)
    end_sample = librosa.time_to_samples(onset_times[index+1], sr=sr)

    segment = y[start_sample:end_sample]
    
    segment, index = librosa.effects.trim(segment, top_db=2)

    # -- Create Spectrogram --
    mel_spect = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128, fmin=f_min, fmax=f_max)

    librosa.display.specshow(mel_spect, y_axis='mel', x_axis='time', fmin=f_min, fmax=f_max)
    fig = plt.gcf()
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')

    spectpath = "spect\\spectrogram.png"
    # c += 1
    fig.savefig(spectpath)

    fig.clear()


    # -- Resize Image --
    img = cv2.imread(spectpath)
    x1, y1, x2, y2 = 80, 59, 477, 426

    cropped = img[y1:y2, x1:x2]

    resized = cv2.resize(cropped, (640, 640))


    # -- Predict --
    results = model.predict(source=resized, stream=True)

    for result in results:
        probs = result.probs
        top1 = probs.top1
        conf = probs.top1conf

        print(top1, conf)

        predicteds.append(names[top1])

#print(predicteds)

for i in range(0, 6):
    for line in predicteds:
        line = line[::-1]
        print(class_dict[line[i]], end="-")
    print()