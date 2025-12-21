import os
from pcg_analyzer import PCGAnalyzer
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks, hilbert, butter, filtfilt,  welch
import pywt
from IPython.display import Audio, display

# === Import signal utilities ==
from pcg_analyzer import (
    compute_fft,
    compute_psd
)

def analyze_single_pcg(file_path, output_dir="outputs"):
    """
    Pipeline lengkap analisis satu file PCG
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    analyzer = PCGAnalyzer(file_path)

    # === Processing Pipeline ===
    analyzer.bandpass_filter(lowcut=20, highcut=500)
    analyzer.wavelet_denoise()
    analyzer.extract_envelope()
    analyzer.detect_peaks()

    # === Output ===
    summary = analyzer.generate_summary()
    analyzer.save_processed_signals(output_dir)

    return analyzer, summary


def extract_features(sig, sr):
    print("======================================")
    print("=== FEATURE EXTRACTION PER SIGNAL ====")
    print("======================================")
    print(f"Signal length            : {len(sig)}")
    print(f"Sampling rate (sr)       : {sr}")

    feats = {}

    # TIME DOMAIN FEATURES
    print("\n--- Envelope + Peak Detection ---")
    env = envelope_hilbert(sig)

    peaks, peak_times, props = detect_beats_from_envelope(
        env, sr, distance_ms=250, prominence=0.12
    )
    cycles = segment_cycles(peaks, sr)

    print(f"Envelope mean            : {np.mean(env):.6f}")
    print(f"Envelope std             : {np.std(env):.6f}")
    print(f"Envelope max             : {np.max(env):.6f}")
    print(f"Detected peaks           : {len(peaks)}")

    feats['n_peaks']  = len(peaks)
    feats['env_mean'] = np.mean(env)
    feats['env_std']  = np.std(env)
    feats['env_max']  = np.max(env)

    # Interval features
    if len(peak_times) >= 2:
        intervals = np.diff(peak_times)
        feats['interval_mean']   = np.mean(intervals)
        feats['interval_std']    = np.std(intervals)
        feats['interval_median'] = np.median(intervals)

        print("Intervals (sec):")
        print("  mean   :", feats['interval_mean'])
        print("  std    :", feats['interval_std'])
        print("  median :", feats['interval_median'])
    else:
        feats['interval_mean']   = np.nan
        feats['interval_std']    = np.nan
        feats['interval_median'] = np.nan
        print("Not enough peaks for interval statistics.")

    # FREQUENCY DOMAIN (FFT)
    print("\n--- FFT FEATURES ---")
    xf, yf = compute_fft(sig, sr)

    total_energy = np.sum(yf)
    feats['spec_energy_total'] = total_energy
    print(f"Total spectral energy    : {total_energy:.6f}")

    def band_energy(xf, yf, low, high):
        mask = (xf >= low) & (xf <= high)
        return np.sum(yf[mask])

    feats['energy_0_60']    = band_energy(xf, yf, 0, 60)
    feats['energy_60_120']  = band_energy(xf, yf, 60, 120)
    feats['energy_100_200'] = band_energy(xf, yf, 100, 200)
    feats['energy_200_500'] = band_energy(xf, yf, 200, 500)

    print("Band energies:")
    print(f"  0–60 Hz    : {feats['energy_0_60']:.5f}")
    print(f"  60–120 Hz  : {feats['energy_60_120']:.5f}")
    print(f"  100–200 Hz : {feats['energy_100_200']:.5f}")
    print(f"  200–500 Hz : {feats['energy_200_500']:.5f}")

    # SPECTRAL ENTROPY (PSD)
    print("\n--- PSD + Spectral Entropy ---")
    psd_f, psd = compute_psd(sig, sr)

    psd_sum = np.sum(psd)
    print(f"PSD total power          : {psd_sum:.6f}")

    psd_norm = psd / psd_sum if psd_sum > 0 else psd
    psd_norm = np.where(psd_norm <= 1e-12, 1e-12, psd_norm)

    entropy = -np.sum(psd_norm * np.log2(psd_norm))
    feats['spec_entropy'] = entropy
    print(f"Spectral entropy         : {entropy:.6f}")

    # MFCC FEATURES
    print("\n--- MFCC FEATURES (13 Coeff, using only #0) ---")
    mfcc = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=13)

    feats['mfcc_mean_0'] = np.mean(mfcc[0])
    feats['mfcc_std_0']  = np.std(mfcc[0])

    print(f"MFCC[0] mean             : {feats['mfcc_mean_0']:.6f}")
    print(f"MFCC[0] std              : {feats['mfcc_std_0']:.6f}")

    print("\n=== Feature Extraction Completed ===\n")

    return feats, env, peaks


# Blok 3 - envelope & simple S1/S2 detection

def envelope_hilbert(sig):
    print("=== ENVELOPE (HILBERT TRANSFORM) ===")
    print(f"Input signal length       : {len(sig)}")

    analytic = hilbert(sig)
    env = np.abs(analytic)

    print(f"Envelope min/max/mean     : {env.min():.6f} / {env.max():.6f} / {env.mean():.6f}\n")
    return env


def detect_beats_from_envelope(env, sr, distance_ms=250, prominence=0.2):
    print("=== PEAK DETECTION (S1/S2) ===")
    print(f"Sampling rate (sr)        : {sr}")
    print(f"distance_ms               : {distance_ms} ms")
    print(f"prominence factor         : {prominence}")

    distance = int(sr * distance_ms / 1000)
    threshold_prominence = prominence * np.max(env)

    print(f"Computed distance (samples): {distance}")
    print(f"Max envelope value         : {np.max(env):.6f}")
    print(f"Prominence threshold       : {threshold_prominence:.6f}")

    peaks, props = find_peaks(env, distance=distance, prominence=threshold_prominence)

    print(f"Total peaks detected       : {len(peaks)}")
    if len(peaks) > 0:
        print(f"Peak indices (first 10)    : {peaks[:10]}")
        print(f"Peak times (sec, first 10) : {(peaks[:10] / sr).round(4)}")

    peak_times = peaks / sr
    print("")  # line break
    return peaks, peak_times, props


def segment_cycles(peaks, sr):
    print("=== SIMPLE CYCLE SEGMENTATION (S1 → S2) ===")
    print(f"Total detected peaks       : {len(peaks)}")

    cycles = []
    for i in range(len(peaks)-1):
        start = peaks[i]
        end = peaks[i+1]
        duration_s = (end - start) / sr
        cycles.append((start, end))

        print(f"Cycle {i+1}: start={start}, end={end}, duration={duration_s:.4f} sec")

    print(f"Total cycles formed        : {len(cycles)}\n")
    return cycles




# Blok 2 - bandpass + wavelet denoising + resample util
def bandpass_filter(sig, sr, low=20, high=500, order=4):
    nyq = 0.5 * sr
    lown = low/nyq
    highn = high/nyq

    print("=== BANDPASS FILTER ===")
    print(f"Sampling rate (sr)        : {sr}")
    print(f"Nyquist frequency (nyq)   : {nyq}")
    print(f"Low cutoff (normalized)   : {lown}")
    print(f"High cutoff (normalized)  : {highn}")
    print(f"Filter order              : {order}")

    b, a = butter(order, [lown := lown, highn], btype='band')

    print(f"Filter coefficients b     : {b}")
    print(f"Filter coefficients a     : {a}")

    filtered = filtfilt(b, a, sig)
    print(f"Output signal length      : {len(filtered)}\n")
    return filtered


def wavelet_denoise(sig, wavelet='db8', level=None):
    print("=== WAVELET DENOISING ===")
    print(f"Wavelet used              : {wavelet}")

    coeffs = pywt.wavedec(sig, wavelet, mode='symmetric')
    print(f"Total decomposition levels: {len(coeffs)-1}")

    if level is None:
        level = min(len(coeffs)-1, 5)
    print(f"Selected threshold level  : {level}")

    sigma = (1/0.6745) * np.median(np.abs(coeffs[-level]))
    print(f"Estimated noise sigma     : {sigma}")

    uthresh = sigma * np.sqrt(2 * np.log(len(sig)))
    print(f"Universal threshold       : {uthresh}")

    denoised_coeffs = []
    for i, c in enumerate(coeffs):
        if i > 0:
            c2 = pywt.threshold(c, value=uthresh, mode='soft')
            print(f"Coeff {i} | before: {np.mean(np.abs(c)):.6f}  after: {np.mean(np.abs(c2)):.6f}")
            denoised_coeffs.append(c2)
        else:
            denoised_coeffs.append(c)

    reconstructed = pywt.waverec(denoised_coeffs, wavelet, mode='symmetric')
    print(f"Reconstructed signal len  : {len(reconstructed)}\n")
    return reconstructed


def ensure_sr(sig, sr, target_sr=2000):
    print("=== RESAMPLING CHECK ===")
    print(f"Original SR               : {sr}")
    print(f"Target SR                 : {target_sr}")

    if sr == target_sr:
        print("SR sama → tidak di-resample.\n")
        return sig, sr

    sig_resampled = librosa.resample(sig, orig_sr=sr, target_sr=target_sr)
    print(f"Resampled signal length   : {len(sig_resampled)}\n")
    return sig_resampled, target_sr

def analyze_and_visualize(file_path, label):
    print(f"\n=== {label} ===")
    print("Path file:", file_path)

    y, sr = librosa.load(file_path, sr=None)

    plt.figure(figsize=(12, 3))
    t = np.arange(len(y)) / sr
    plt.plot(t, y)
    plt.title(f"Waveform - {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


    # Insight: FFT menunjukkan frekuensi dominan. Murmur biasanya tampak sebagai energi frekuensi rendah–menengah yang melebar.
    Y = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1/sr)

    plt.figure(figsize=(12, 3))
    plt.plot(freqs, Y)
    plt.title(f"FFT Spectrum - {label}")
    plt.xlim(0, 500)
    plt.xlabel("Hz")
    plt.show()

    # Insight: PSD (Welch) memberi estimasi energi rata-rata per frekuensi. Cocok untuk membedakan murmur AS, MR, MS, dll berdasarkan distribusi power.
    f_welch, Pxx = welch(y, fs=sr, nperseg=1024)

    plt.figure(figsize=(12, 3))
    plt.semilogy(f_welch, Pxx)
    plt.title(f"PSD - {label}")
    plt.xlim(0, 500)
    plt.xlabel("Hz")
    plt.show()

    # Insight: Spectrogram STFT menampilkan perubahan energi frekuensi terhadap waktu. S1 & S2 terlihat jelas, murmur tampak sebagai pola gelap yang memanjang.
    S = np.abs(librosa.stft(y))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.title(f"Spectrogram (STFT) - {label}")
    plt.ylim(0, 500)
    plt.show()

    # Insight: CWT Scalogram (Wavelet) sangat baik untuk mendeteksi murmur transien/dinamis. Resolusi waktu lebih halus dibanding STFT.
    import pywt
    scales = np.arange(1, 128)
    cwtmatr, freqs_cwt = pywt.cwt(y, scales, 'morl', sampling_period=1/sr)

    plt.figure(figsize=(12, 4))
    plt.imshow(np.abs(cwtmatr), extent=[0, len(y)/sr, freqs_cwt[-1], freqs_cwt[0]],
               cmap='jet', aspect='auto')
    plt.title(f"CWT Scalogram - {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(0, 500)
    plt.colorbar()
    plt.show()

    print("Playing audio:")
    display(Audio(y, rate=sr))
