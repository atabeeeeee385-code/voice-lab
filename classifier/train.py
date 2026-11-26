"""Train a spoken-digit classifier on the Free Spoken Digit Dataset (FSDD).

Evaluation is speaker-independent: the model is always tested on a speaker
it never saw in training (leave-one-speaker-out).
"""
import sys, os, glob, json, pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from features import load_wav, clip_features

DATA = sys.argv[1] if len(sys.argv) > 1 else "fsdd/recordings"
OUT = os.path.dirname(os.path.abspath(__file__))


def load():
    X, y, spk = [], [], []
    for p in sorted(glob.glob(os.path.join(DATA, "*.wav"))):
        d, s, _ = os.path.basename(p)[:-4].split("_")
        X.append(clip_features(load_wav(p)))
        y.append(int(d)); spk.append(s)
    return np.array(X), np.array(y), np.array(spk)


def model():
    return make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", kernel="rbf"))


if __name__ == "__main__":
    X, y, spk = load()
    print("clips:", len(y), "features:", X.shape[1])
    accs, y_true, y_pred = {}, [], []
    for s in sorted(set(spk)):
        te = spk == s
        m = model().fit(X[~te], y[~te])
        p = m.predict(X[te])
        accs[s] = accuracy_score(y[te], p)
        y_true += list(y[te]); y_pred += list(p)
        print(f"held-out speaker {s:10s} acc = {accs[s]:.3f}")
    overall = accuracy_score(y_true, y_pred)
    print(f"overall leave-one-speaker-out accuracy: {overall:.3f}")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("predicted digit"); ax.set_ylabel("spoken digit")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_title(f"Confusion matrix (unseen speakers), acc {overall:.1%}")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "..", "docs", "confusion_matrix.png"), dpi=140)
    final = model().fit(X, y)
    with open(os.path.join(OUT, "model.pkl"), "wb") as f:
        pickle.dump(final, f)
    json.dump({"overall": overall, "per_speaker": accs}, open(os.path.join(OUT, "results.json"), "w"), indent=2)
