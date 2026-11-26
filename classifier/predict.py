"""Predict the spoken digit (0-9) in one or more WAV files.
    python predict.py my_recording.wav [more.wav ...]
"""
import sys, os, pickle
import numpy as np
from features import load_wav, clip_features

here = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(here, "model.pkl"), "rb"))

for path in sys.argv[1:]:
    x = clip_features(load_wav(path))[None, :]
    digit = int(model.predict(x)[0])
    scores = model.decision_function(x)[0]
    top = np.argsort(scores)[::-1][:3]
    print(f"{os.path.basename(path)}: digit {digit}   (runner-ups: {', '.join(str(int(t)) for t in top[1:])})")
