# Spoken digit classifier (MFCC + SVM)

A small trained model that listens to a short recording of someone saying a digit (0 to 9) and tells you which one.

## Results
Trained on the open [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset) (3,000 recordings, 6 speakers, 8 kHz).

Tested the honest way: **leave-one-speaker-out**. The model is trained on 5 speakers and tested on the 6th, which it never heard. Repeated for each speaker.

| Held-out speaker | Accuracy |
|---|---|
| george | 65.8% |
| jackson | 87.8% |
| lucas | 90.2% |
| nicolas | 68.8% |
| theo | 90.8% |
| yweweler | 81.4% |
| **overall** | **80.8%** (chance = 10%) |

See `../docs/confusion_matrix.png`.

## How it works
1. Trim silence, pre-emphasis, 32 ms Hamming frames every 10 ms.
2. Mel filterbank, log, DCT: 13 MFCCs per frame (written in NumPy, no audio library).
3. Clip features: mean and std of the MFCCs, plus mean MFCCs of 5 time segments (keeps a rough word shape), ZCR and duration: 93 numbers per clip.
4. Standardise, then an RBF-kernel SVM (scikit-learn).

## Run
```bash
pip install numpy scipy scikit-learn matplotlib
git clone https://github.com/Jakobovski/free-spoken-digit-dataset fsdd
python train.py fsdd/recordings      # evaluates + saves model.pkl
python predict.py my_recording.wav   # one digit per WAV file, English
```

## Limits (be honest)
- Only 6 speakers, studio-quiet recordings, English digits. Accuracy on your own voice and a phone mic can be lower than 80%.
- A weak speaker can drop to about 66%. More speakers would help most.
- Isolated digits only, not continuous speech.
- A stronger next step is a small CNN on the spectrogram; I tried a few simple alternatives (time-grid features, MLP) and they were worse than this SVM.
