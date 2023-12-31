# TabMLProject

Utilizes a Yolov8 custom trained model that has been trained on a dataset consisting of spectrograms generated from the first 3 guitar frets for each string (open, first, second, third).

The onsets are detected, and a spectrogram is created for the space between each onset. The spectrogram is then processed and given to the model to produce a prediction for which note's spectrogram it sees.

Class names are written such that they correspond to how the strings would be played on the instrument, with different letters corresponding to certain numbers.
For example, the class name for an open low-E string is wxxxxx, where w corresponds to 0 and x corresponds to unplayed strings. These class names are "decoded" after all predictions have been made to produce the familiar tablature structure. The resulting string for this example would be 0-----.

Given enough time, the dataset could grow to contain spectrograms of every string and chord, though this would be a monumental task.

The current audio file (audio\Pentatonic.wav) is a simple recording of the pentatonic scale. It produces the following output:

--------------------0-3-

----------------0-0-----

------------0-2---------

--------0-2-------------

----0-2-----------------

0-3---------------------

Note that the model misinterpereted the third-to-last note as an open B string, where it should have been third fret B string. This may be due to a class imbalance in the training data, which should be addressed.
