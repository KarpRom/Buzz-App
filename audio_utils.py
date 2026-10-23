"""Audio loading and spectrogram rendering shared by the Streamlit apps."""

import io

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def load_audio(audio_path, duration=5):
    """Load audio at its native sample rate; keep the first `duration` seconds (None = all)."""
    audio, sr = librosa.load(audio_path, sr=None)
    if duration is not None:
        audio = audio[: int(duration * sr)]
    return audio, sr


def create_spectrogram(audio, sr, n_fft, hop_length, db_range, mel=False, n_mels=128,
                       fmax=None, title="Audio Spectrogram", figsize=(10, 4)):
    """Spectrogram as a PNG buffer: linear-frequency STFT, or mel-scaled if `mel`."""
    fig, ax = plt.subplots(figsize=figsize)
    if mel:
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                           n_mels=n_mels, fmax=fmax)
        D = librosa.power_to_db(S, ref=np.max)
        y_axis = "mel"
    else:
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
        y_axis = "hz"
    vmax = np.max(D)
    img = librosa.display.specshow(D, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis,
                                   ax=ax, vmin=vmax - db_range, vmax=vmax, cmap="viridis", fmax=fmax)
    if fmax and not mel:
        ax.set_ylim(0, fmax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    cbar = plt.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.set_label("Power (dB)")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf
