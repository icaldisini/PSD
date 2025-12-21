import librosa
import numpy as np
import pywt
from scipy.signal import butter, filtfilt
from IPython.display import Audio, display

# =========================================================
# DSP CORE (SAFE)
# =========================================================
def bandpass_filter(x, sr, low=20, high=500, order=3):
    nyq = sr / 2
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    y = filtfilt(b, a, x)
    y = np.nan_to_num(y)
    y = y / (np.max(np.abs(y)) + 1e-8)
    return y.astype(np.float32)

def wavelet_denoise(x, wavelet="db8", level=5):
    coeffs = pywt.wavedec(x, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeffs = [pywt.threshold(c, uthresh, mode="soft") for c in coeffs]
    y = pywt.waverec(coeffs, wavelet)
    y = y[:len(x)]
    y = np.nan_to_num(y)
    y = y / (np.max(np.abs(y)) + 1e-8)
    return y.astype(np.float32)

# =========================================================
# ONE CALL PLAYER
# =========================================================
def play_pcg_sample(path):
    y, sr = librosa.load(path, sr=None, mono=True)

    y_bp = bandpass_filter(y, sr)
    y_dn = wavelet_denoise(y_bp)

    print("🎧 ORIGINAL")
    display(Audio(y, rate=sr))

    print("🎧 BANDPASS (20–500 Hz)")
    display(Audio(y_bp, rate=sr))

    print("🎧 WAVELET DENOISED")
    display(Audio(y_dn, rate=sr))
