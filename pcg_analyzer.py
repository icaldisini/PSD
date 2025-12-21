import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pywt
import os
from IPython.display import Audio, display
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.fft import rfft, rfftfreq
import pywt
import librosa

class PCGAnalyzer:
    """
    Kelas untuk analisis sinyal PCG (Phonocardiogram)
    Melakukan preprocessing dan analisis time-domain
    """

    def __init__(self, file_path, target_sr=2000):
        """
        Inisialisasi PCG Analyzer

        Parameters:
        -----------
        file_path : str
            Path ke file audio PCG (.wav)
        target_sr : int
            Target sampling rate (default: 2000 Hz)
        """
        self.file_path = file_path
        self.target_sr = target_sr
        self.original_signal = None
        self.original_sr = None
        self.filtered_signal = None
        self.denoised_signal = None
        self.envelope = None
        self.envelope_smooth = None
        self.peaks = None
        self.s1_peaks = None
        self.s2_peaks = None

        # Load audio file
        self._load_audio()

    def _load_audio(self):
        """Load dan resample audio file"""
        try:
            sr, data = wavfile.read(self.file_path)

            # Konversi ke mono jika stereo
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # Normalisasi ke range [-1, 1]
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            else:
                data = data.astype(np.float32)

            self.original_sr = sr

            # Resample jika perlu
            if sr != self.target_sr:
                num_samples = int(len(data) * self.target_sr / sr)
                self.original_signal = signal.resample(data, num_samples)
            else:
                self.original_signal = data

            print(f"✓ Audio loaded successfully")
            print(f"  - Original sampling rate: {sr} Hz")
            print(f"  - Target sampling rate: {self.target_sr} Hz")
            print(f"  - Duration: {len(self.original_signal)/self.target_sr:.2f} seconds")
            print(f"  - Samples: {len(self.original_signal)}")

        except Exception as e:
            print(f"✗ Error loading audio: {e}")
            raise


def bandpass_filter(self, lowcut=20, highcut=500, order=4, show_plot=True):
    """
    Step 1: Bandpass Filtering (20-500 Hz)

    Parameters:
    -----------
    lowcut : float
        Frekuensi cutoff bawah (Hz)
    highcut : float
        Frekuensi cutoff atas (Hz)
    order : int
        Order filter (semakin tinggi = semakin tajam)
    show_plot : bool
        Tampilkan visualisasi hasil filtering
    """
    print("\n" + "="*70)
    print("STEP 1: BANDPASS FILTERING")
    print("="*70)

    # Desain Butterworth bandpass filter
    nyquist = 0.5 * self.target_sr
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = signal.butter(order, [low, high], btype='band')

    # Apply filter
    self.filtered_signal = signal.filtfilt(b, a, self.original_signal)

    print(f"✓ Bandpass filter applied: {lowcut}-{highcut} Hz (Order: {order})")
    print(f"  - Signal length: {len(self.filtered_signal)} samples")

    if show_plot:
        self._plot_bandpass_results(lowcut, highcut)

    return self.filtered_signal

def _plot_bandpass_results(self, lowcut, highcut):
    """Visualisasi hasil bandpass filtering"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Step 1: Bandpass Filtering Results', fontsize=16, fontweight='bold')

    time = np.arange(len(self.original_signal)) / self.target_sr

    # Plot 1: Original Signal
    axes[0, 0].plot(time, self.original_signal, linewidth=0.5, color='blue', alpha=0.7)
    axes[0, 0].set_title('Original Signal (Time Domain)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Filtered Signal
    axes[0, 1].plot(time, self.filtered_signal, linewidth=0.5, color='red', alpha=0.7)
    axes[0, 1].set_title('Filtered Signal (Time Domain)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Comparison (Zoomed)
    zoom_samples = min(int(2 * self.target_sr), len(self.original_signal))
    time_zoom = time[:zoom_samples]
    axes[1, 0].plot(time_zoom, self.original_signal[:zoom_samples],
                   label='Original', linewidth=1, alpha=0.7)
    axes[1, 0].plot(time_zoom, self.filtered_signal[:zoom_samples],
                   label='Filtered', linewidth=1, alpha=0.7)
    axes[1, 0].set_title('Comparison (Zoomed - First 2 seconds)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Frequency Spectrum (Original)
    freqs_orig, psd_orig = signal.welch(self.original_signal, self.target_sr, nperseg=1024)
    axes[1, 1].semilogy(freqs_orig, psd_orig, linewidth=1.5, color='blue')
    axes[1, 1].axvline(lowcut, color='green', linestyle='--', label=f'Lowcut: {lowcut} Hz')
    axes[1, 1].axvline(highcut, color='orange', linestyle='--', label=f'Highcut: {highcut} Hz')
    axes[1, 1].set_title('Original Signal - Frequency Spectrum')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Power Spectral Density')
    axes[1, 1].set_xlim([0, 600])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 5: Frequency Spectrum (Filtered)
    freqs_filt, psd_filt = signal.welch(self.filtered_signal, self.target_sr, nperseg=1024)
    axes[2, 0].semilogy(freqs_filt, psd_filt, linewidth=1.5, color='red')
    axes[2, 0].axvline(lowcut, color='green', linestyle='--', label=f'Lowcut: {lowcut} Hz')
    axes[2, 0].axvline(highcut, color='orange', linestyle='--', label=f'Highcut: {highcut} Hz')
    axes[2, 0].set_title('Filtered Signal - Frequency Spectrum')
    axes[2, 0].set_xlabel('Frequency (Hz)')
    axes[2, 0].set_ylabel('Power Spectral Density')
    axes[2, 0].set_xlim([0, 600])
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 6: Comparison
    axes[2, 1].semilogy(freqs_orig, psd_orig, label='Original', linewidth=1.5, alpha=0.7, color='blue')
    axes[2, 1].semilogy(freqs_filt, psd_filt, label='Filtered', linewidth=1.5, alpha=0.7, color='red')
    axes[2, 1].axvline(lowcut, color='green', linestyle='--', alpha=0.5)
    axes[2, 1].axvline(highcut, color='orange', linestyle='--', alpha=0.5)
    axes[2, 1].set_title('Frequency Spectrum Comparison')
    axes[2, 1].set_xlabel('Frequency (Hz)')
    axes[2, 1].set_ylabel('Power Spectral Density')
    axes[2, 1].set_xlim([0, 600])
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    print("✓ Visualization displayed!")

# Tambahkan method ke class
PCGAnalyzer.bandpass_filter = bandpass_filter
PCGAnalyzer._plot_bandpass_results = _plot_bandpass_results

print("✓ Step 1: Bandpass Filtering - Ready!")


def wavelet_denoise(self, wavelet='db6', level=6, mode='soft', show_plot=True):
    """
    Step 2: Wavelet Denoising

    Parameters:
    -----------
    wavelet : str
        Jenis wavelet (db6, db4, sym5, coif5)
    level : int
        Level dekomposisi wavelet
    mode : str
        Mode thresholding ('soft' atau 'hard')
    show_plot : bool
        Tampilkan visualisasi hasil denoising
    """
    print("\n" + "="*70)
    print("STEP 2: WAVELET DENOISING")
    print("="*70)

    if self.filtered_signal is None:
        print("⚠ Warning: Filtered signal not found. Using original signal.")
        input_signal = self.original_signal
    else:
        input_signal = self.filtered_signal

    # Decompose signal
    coeffs = pywt.wavedec(input_signal, wavelet, level=level)

    # Hitung threshold
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(input_signal)))

    # Apply thresholding
    coeffs_thresh = [coeffs[0]]
    for i in range(1, len(coeffs)):
        if mode == 'soft':
            coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
        else:
            coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='hard'))

    # Reconstruct signal
    self.denoised_signal = pywt.waverec(coeffs_thresh, wavelet)

    # Pastikan panjang sama
    if len(self.denoised_signal) > len(input_signal):
        self.denoised_signal = self.denoised_signal[:len(input_signal)]
    elif len(self.denoised_signal) < len(input_signal):
        self.denoised_signal = np.pad(self.denoised_signal,
                                     (0, len(input_signal) - len(self.denoised_signal)))

    # Hitung SNR
    noise_removed = input_signal - self.denoised_signal
    snr_before = 10 * np.log10(np.var(input_signal) / np.var(noise_removed))

    print(f"✓ Wavelet denoising applied")
    print(f"  - Wavelet: {wavelet}")
    print(f"  - Level: {level}")
    print(f"  - Mode: {mode}")
    print(f"  - Threshold: {threshold:.4f}")
    print(f"  - SNR improvement: {snr_before:.2f} dB")

    if show_plot:
        self._plot_wavelet_results(coeffs, coeffs_thresh, wavelet, level)

    return self.denoised_signal

def _plot_wavelet_results(self, coeffs, coeffs_thresh, wavelet, level):
    """Visualisasi hasil wavelet denoising"""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Step 2: Wavelet Denoising Results', fontsize=16, fontweight='bold')

    time = np.arange(len(self.filtered_signal)) / self.target_sr

    # Plot 1: Input Signal
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(time, self.filtered_signal, linewidth=0.5, color='blue')
    ax1.set_title('Input: Filtered Signal')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Denoised Signal
    ax2 = plt.subplot(4, 2, 2)
    ax2.plot(time, self.denoised_signal, linewidth=0.5, color='red')
    ax2.set_title('Output: Denoised Signal')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Comparison (Zoomed)
    ax3 = plt.subplot(4, 2, 3)
    zoom_samples = min(int(2 * self.target_sr), len(self.filtered_signal))
    time_zoom = time[:zoom_samples]
    ax3.plot(time_zoom, self.filtered_signal[:zoom_samples],
            label='Before', linewidth=1, alpha=0.7)
    ax3.plot(time_zoom, self.denoised_signal[:zoom_samples],
            label='After', linewidth=1, alpha=0.7)
    ax3.set_title('Comparison (Zoomed - First 2s)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Removed Noise
    ax4 = plt.subplot(4, 2, 4)
    noise = self.filtered_signal - self.denoised_signal
    ax4.plot(time, noise, linewidth=0.5, color='green', alpha=0.7)
    ax4.set_title('Removed Noise')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Coefficients (Before)
    ax5 = plt.subplot(4, 2, 5)
    for i, coeff in enumerate(coeffs):
        ax5.plot(coeff + i*2, label=f'L{i}', linewidth=0.5, alpha=0.7)
    ax5.set_title(f'Coefficients (Before) - {wavelet}')
    ax5.set_xlabel('Index')
    ax5.set_ylabel('Value')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Coefficients (After)
    ax6 = plt.subplot(4, 2, 6)
    for i, coeff in enumerate(coeffs_thresh):
        ax6.plot(coeff + i*2, label=f'L{i}', linewidth=0.5, alpha=0.7)
    ax6.set_title('Coefficients (After)')
    ax6.set_xlabel('Index')
    ax6.set_ylabel('Value')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # Plot 7: Frequency Spectrum
    ax7 = plt.subplot(4, 2, 7)
    freqs_b, psd_b = signal.welch(self.filtered_signal, self.target_sr, nperseg=1024)
    freqs_a, psd_a = signal.welch(self.denoised_signal, self.target_sr, nperseg=1024)
    ax7.semilogy(freqs_b, psd_b, label='Before', linewidth=1.5, alpha=0.7)
    ax7.semilogy(freqs_a, psd_a, label='After', linewidth=1.5, alpha=0.7)
    ax7.set_title('Frequency Spectrum')
    ax7.set_xlabel('Frequency (Hz)')
    ax7.set_ylabel('PSD')
    ax7.set_xlim([0, 500])
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Plot 8: Statistics
    ax8 = plt.subplot(4, 2, 8)
    ax8.axis('off')
    stats_text = f"""
    WAVELET DENOISING STATISTICS
    {'='*40}
    Wavelet: {wavelet}
    Level: {level}

    RMS:
    - Input: {np.sqrt(np.mean(self.filtered_signal**2)):.4f}
    - Output: {np.sqrt(np.mean(self.denoised_signal**2)):.4f}
    - Noise: {np.sqrt(np.mean(noise**2)):.4f}

    Energy Retained: {(np.sum(self.denoised_signal**2)/np.sum(self.filtered_signal**2)*100):.2f}%

    Sparsity: {(1 - sum(np.count_nonzero(c) for c in coeffs_thresh)/sum(len(c) for c in coeffs))*100:.2f}%
    """
    ax8.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', va='center')

    plt.tight_layout()
    plt.show()
    print("✓ Visualization displayed!")

# Tambahkan method ke class
PCGAnalyzer.wavelet_denoise = wavelet_denoise
PCGAnalyzer._plot_wavelet_results = _plot_wavelet_results

print("✓ Step 2: Wavelet Denoising - Ready!")


def extract_envelope(self, show_plot=True):
    """
    Step 3: Envelope Extraction (Hilbert Transform)
    """
    print("\n" + "="*70)
    print("STEP 3: ENVELOPE EXTRACTION (HILBERT TRANSFORM)")
    print("="*70)

    if self.denoised_signal is None:
        print("⚠ Warning: Using filtered signal")
        input_signal = self.filtered_signal if self.filtered_signal is not None else self.original_signal
    else:
        input_signal = self.denoised_signal

    # Hilbert transform
    analytic_signal = signal.hilbert(input_signal)
    self.envelope = np.abs(analytic_signal)

    # Smooth envelope
    window_size = int(0.02 * self.target_sr)  # 20ms
    if window_size % 2 == 0:
        window_size += 1
    self.envelope_smooth = signal.savgol_filter(self.envelope, window_size, 3)

    # Instantaneous frequency
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * self.target_sr

    print(f"✓ Envelope extracted")
    print(f"  - Mean: {np.mean(self.envelope):.4f}")
    print(f"  - Std: {np.std(self.envelope):.4f}")
    print(f"  - Max: {np.max(self.envelope):.4f}")

    if show_plot:
        self._plot_envelope_results(input_signal, instantaneous_frequency)

    return self.envelope

def _plot_envelope_results(self, input_signal, inst_freq):
    """Visualisasi envelope extraction"""
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle('Step 3: Envelope Extraction', fontsize=16, fontweight='bold')

    time = np.arange(len(input_signal)) / self.target_sr
    time_freq = np.arange(len(inst_freq)) / self.target_sr

    # Plot 1: Signal with Envelope
    axes[0, 0].plot(time, input_signal, linewidth=0.5, alpha=0.7, label='Signal')
    axes[0, 0].plot(time, self.envelope, linewidth=1.5, color='red', label='Envelope', alpha=0.8)
    axes[0, 0].plot(time, -self.envelope, linewidth=1.5, color='red', alpha=0.8)
    axes[0, 0].set_title('Signal with Envelope')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Smoothed Envelope
    axes[0, 1].plot(time, self.envelope, linewidth=1, alpha=0.5, label='Raw')
    axes[0, 1].plot(time, self.envelope_smooth, linewidth=2, color='red', label='Smoothed')
    axes[0, 1].set_title('Envelope Smoothing')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Zoomed
    zoom_samples = min(int(2 * self.target_sr), len(input_signal))
    time_zoom = time[:zoom_samples]
    axes[1, 0].plot(time_zoom, input_signal[:zoom_samples], linewidth=0.7, alpha=0.7)
    axes[1, 0].plot(time_zoom, self.envelope_smooth[:zoom_samples], linewidth=2, color='red')
    axes[1, 0].set_title('Zoomed - First 2s')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Instantaneous Frequency
    axes[1, 1].plot(time_freq, inst_freq, linewidth=0.5, alpha=0.7)
    axes[1, 1].set_title('Instantaneous Frequency')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Frequency (Hz)')
    axes[1, 1].set_ylim([0, 500])
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 5: Distribution
    axes[2, 0].hist(self.envelope, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[2, 0].axvline(np.mean(self.envelope), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(self.envelope):.4f}')
    axes[2, 0].set_title('Amplitude Distribution')
    axes[2, 0].set_xlabel('Amplitude')
    axes[2, 0].set_ylabel('Count')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 6: Envelope Spectrum
    freqs_env, psd_env = signal.welch(self.envelope_smooth, self.target_sr, nperseg=1024)
    axes[2, 1].semilogy(freqs_env, psd_env, linewidth=1.5, color='red')
    axes[2, 1].set_title('Envelope Spectrum')
    axes[2, 1].set_xlabel('Frequency (Hz)')
    axes[2, 1].set_ylabel('PSD')
    axes[2, 1].set_xlim([0, 50])
    axes[2, 1].grid(True, alpha=0.3)

    # Plot 7: Energy Contour
    segment_duration = 0.1
    segment_samples = int(segment_duration * self.target_sr)
    n_segments = len(self.envelope_smooth) // segment_samples

    energy_segments = []
    time_segments = []
    for i in range(n_segments):
        start = i * segment_samples
        end = start + segment_samples
        energy_segments.append(np.mean(self.envelope_smooth[start:end]))
        time_segments.append(i * segment_duration)

    axes[3, 0].bar(time_segments, energy_segments, width=segment_duration*0.8,
                  alpha=0.7, color='orange', edgecolor='black')
    axes[3, 0].set_title('Energy Contour (100ms segments)')
    axes[3, 0].set_xlabel('Time (s)')
    axes[3, 0].set_ylabel('Energy')
    axes[3, 0].grid(True, alpha=0.3, axis='y')

    # Plot 8: Statistics
    axes[3, 1].axis('off')
    envelope_peaks = signal.find_peaks(self.envelope_smooth, distance=int(0.3*self.target_sr))[0]
    if len(envelope_peaks) > 1:
        peak_intervals = np.diff(envelope_peaks) / self.target_sr
        hr = 60 / np.mean(peak_intervals)
    else:
        hr = 0

    stats_text = f"""
    ENVELOPE STATISTICS
    {'='*40}
    Amplitude:
    - Mean: {np.mean(self.envelope):.4f}
    - Std: {np.std(self.envelope):.4f}
    - Max: {np.max(self.envelope):.4f}

    Energy:
    - Total: {np.sum(self.envelope**2):.2f}
    - RMS: {np.sqrt(np.mean(self.envelope**2)):.4f}

    Temporal:
    - Peaks: {len(envelope_peaks)}
    - Est. HR: {hr:.1f} BPM
    """
    axes[3, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace', va='center')

    plt.tight_layout()
    plt.show()
    print("✓ Visualization displayed!")

# Tambahkan method ke class
PCGAnalyzer.extract_envelope = extract_envelope
PCGAnalyzer._plot_envelope_results = _plot_envelope_results

print("✓ Step 3: Envelope Extraction - Ready!")




def detect_peaks(self, min_distance=0.3, threshold_factor=0.3, show_plot=True):
    """
    Step 4: Peak Detection (S1-S2 Segmentation)

    Parameters:
    -----------
    min_distance : float
        Minimum distance between peaks (seconds)
    threshold_factor : float
        Threshold factor (0-1)
    show_plot : bool
        Tampilkan visualisasi
    """
    print("\n" + "="*70)
    print("STEP 4: PEAK DETECTION (S1-S2 SEGMENTATION)")
    print("="*70)

    if self.envelope is None:
        print("⚠ Extracting envelope first...")
        self.extract_envelope(show_plot=False)

    envelope_to_use = self.envelope_smooth

    # Threshold
    threshold = np.mean(envelope_to_use) + threshold_factor * np.std(envelope_to_use)

    # Detect peaks
    min_distance_samples = int(min_distance * self.target_sr)
    peaks, properties = signal.find_peaks(
        envelope_to_use,
        height=threshold,
        distance=min_distance_samples,
        prominence=np.std(envelope_to_use) * 0.2
    )

    self.peaks = peaks

    # Klasifikasi S1 dan S2
    if len(peaks) > 0:
        s1_peaks = []
        s2_peaks = []

        for i, peak in enumerate(peaks):
            if i % 2 == 0:
                s1_peaks.append(peak)
            else:
                s2_peaks.append(peak)

        self.s1_peaks = np.array(s1_peaks)
        self.s2_peaks = np.array(s2_peaks)

        # Heart rate
        if len(s1_peaks) > 1:
            s1_intervals = np.diff(self.s1_peaks) / self.target_sr
            heart_rate = 60 / np.mean(s1_intervals)
        else:
            heart_rate = 0

        # Systolic & Diastolic
        systolic_intervals = []
        diastolic_intervals = []

        for i in range(min(len(self.s1_peaks), len(self.s2_peaks))):
            systolic = (self.s2_peaks[i] - self.s1_peaks[i]) / self.target_sr
            systolic_intervals.append(systolic)

            if i < len(self.s1_peaks) - 1:
                diastolic = (self.s1_peaks[i+1] - self.s2_peaks[i]) / self.target_sr
                diastolic_intervals.append(diastolic)

        print(f"✓ Peak detection completed")
        print(f"  - Total peaks: {len(peaks)}")
        print(f"  - S1 peaks: {len(self.s1_peaks)}")
        print(f"  - S2 peaks: {len(self.s2_peaks)}")
        print(f"  - Heart rate: {heart_rate:.1f} BPM")
        print(f"  - Mean systolic: {np.mean(systolic_intervals):.3f} s")
        print(f"  - Mean diastolic: {np.mean(diastolic_intervals) if diastolic_intervals else 0:.3f} s")

        if show_plot:
            self._plot_peak_detection_results(envelope_to_use, threshold,
                                             systolic_intervals, diastolic_intervals,
                                             heart_rate)
    else:
        print("⚠ No peaks detected!")
        self.s1_peaks = np.array([])
        self.s2_peaks = np.array([])

    return self.peaks, self.s1_peaks, self.s2_peaks

def _plot_peak_detection_results(self, envelope, threshold, systolic, diastolic, hr):
    """Visualisasi peak detection"""
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle('Step 4: Peak Detection & S1-S2 Segmentation', fontsize=16, fontweight='bold')

    time = np.arange(len(self.denoised_signal)) / self.target_sr

    # Plot 1: Envelope with All Peaks
    axes[0, 0].plot(time, envelope, linewidth=1, color='blue', alpha=0.7, label='Envelope')
    axes[0, 0].axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold')
    axes[0, 0].plot(time[self.peaks], envelope[self.peaks], 'ro', markersize=8, label='Peaks')
    axes[0, 0].set_title('Envelope with Detected Peaks')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: S1 and S2
    axes[0, 1].plot(time, envelope, linewidth=1, color='gray', alpha=0.5)
    axes[0, 1].plot(time[self.s1_peaks], envelope[self.s1_peaks], 'go',
                   markersize=10, label='S1', alpha=0.8)
    axes[0, 1].plot(time[self.s2_peaks], envelope[self.s2_peaks], 'ro',
                   markersize=10, label='S2', alpha=0.8)
    axes[0, 1].set_title('S1 (Green) and S2 (Red)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Signal with Markers
    zoom_samples = min(int(3 * self.target_sr), len(self.denoised_signal))
    time_zoom = time[:zoom_samples]
    axes[1, 0].plot(time_zoom, self.denoised_signal[:zoom_samples], linewidth=0.7, alpha=0.7)

    s1_zoom = self.s1_peaks[self.s1_peaks < zoom_samples]
    s2_zoom = self.s2_peaks[self.s2_peaks < zoom_samples]

    for s1 in s1_zoom:
        axes[1, 0].axvline(time[s1], color='green', linestyle='--', alpha=0.7, linewidth=2)
    for s2 in s2_zoom:
        axes[1, 0].axvline(time[s2], color='red', linestyle='--', alpha=0.7, linewidth=2)

    axes[1, 0].set_title('Signal with S1-S2 Markers (3s)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Cardiac Cycles
    axes[1, 1].plot(time, envelope, linewidth=1, color='blue', alpha=0.7)

    for i in range(min(len(self.s1_peaks), len(self.s2_peaks))):
        systole_start = time[self.s1_peaks[i]]
        systole_end = time[self.s2_peaks[i]]
        axes[1, 1].axvspan(systole_start, systole_end, alpha=0.3, color='red',
                          label='Systole' if i == 0 else '')

        if i < len(self.s1_peaks) - 1:
            diastole_start = time[self.s2_peaks[i]]
            diastole_end = time[self.s1_peaks[i+1]]
            axes[1, 1].axvspan(diastole_start, diastole_end, alpha=0.3, color='blue',
                              label='Diastole' if i == 0 else '')

    axes[1, 1].set_title('Cardiac Cycle Segmentation')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 5: Inter-Peak Intervals
    if len(self.peaks) > 1:
        intervals = np.diff(self.peaks) / self.target_sr
        axes[2, 0].plot(intervals, 'o-', linewidth=2, markersize=6)
        axes[2, 0].axhline(np.mean(intervals), color='red', linestyle='--', linewidth=2)
        axes[2, 0].set_title('Inter-Peak Intervals')
        axes[2, 0].set_xlabel('Interval Index')
        axes[2, 0].set_ylabel('Time (s)')
        axes[2, 0].grid(True, alpha=0.3)

    # Plot 6: Systolic vs Diastolic
    axes[2, 1].bar(['Systolic', 'Diastolic'],
                  [np.mean(systolic), np.mean(diastolic)],
                  color=['red', 'blue'], alpha=0.7, edgecolor='black', linewidth=2)
    axes[2, 1].set_title('Systolic vs Diastolic Duration')
    axes[2, 1].set_ylabel('Duration (s)')
    axes[2, 1].grid(True, alpha=0.3, axis='y')

    for i, (label, value) in enumerate(zip(['Systolic', 'Diastolic'],
                                           [np.mean(systolic), np.mean(diastolic)])):
        axes[2, 1].text(i, value, f'{value:.3f}s', ha='center', va='bottom', fontweight='bold')

    # Plot 7: Heart Rate Variability
    if len(self.s1_peaks) > 1:
        s1_intervals = np.diff(self.s1_peaks) / self.target_sr
        inst_hr = 60 / s1_intervals

        axes[3, 0].plot(inst_hr, 'o-', linewidth=2, markersize=6, color='purple')
        axes[3, 0].axhline(hr, color='red', linestyle='--', linewidth=2, label=f'Mean: {hr:.1f}')
        axes[3, 0].fill_between(range(len(inst_hr)),
                               inst_hr - np.std(inst_hr),
                               inst_hr + np.std(inst_hr),
                               alpha=0.3, color='purple')
        axes[3, 0].set_title('Heart Rate Variability')
        axes[3, 0].set_xlabel('Beat Number')
        axes[3, 0].set_ylabel('HR (BPM)')
        axes[3, 0].legend()
        axes[3, 0].grid(True, alpha=0.3)

    # Plot 8: Statistics
    axes[3, 1].axis('off')

    hrv_sdnn = np.std(np.diff(self.s1_peaks) / self.target_sr) if len(self.s1_peaks) > 1 else 0
    hrv_rmssd = np.sqrt(np.mean(np.diff(np.diff(self.s1_peaks) / self.target_sr)**2)) if len(self.s1_peaks) > 2 else 0

    stats_text = f"""
    PEAK DETECTION STATISTICS
    {'='*40}
    Peaks:
    - Total: {len(self.peaks)}
    - S1: {len(self.s1_peaks)}
    - S2: {len(self.s2_peaks)}

    Cardiac:
    - HR: {hr:.1f} BPM
    - Systolic: {np.mean(systolic):.3f} s
    - Diastolic: {np.mean(diastolic):.3f} s
    - Ratio: {np.mean(systolic)/np.mean(diastolic):.2f}

    HRV:
    - SDNN: {hrv_sdnn*1000:.2f} ms
    - RMSSD: {hrv_rmssd*1000:.2f} ms

    Duration: {len(self.denoised_signal)/self.target_sr:.2f} s
    Cycles: {min(len(self.s1_peaks), len(self.s2_peaks))}
    """
    axes[3, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace', va='center')

    plt.tight_layout()
    plt.show()
    print("✓ Visualization displayed!")

# Tambahkan method ke class
PCGAnalyzer.detect_peaks = detect_peaks
PCGAnalyzer._plot_peak_detection_results = _plot_peak_detection_results

print("✓ Step 4: Peak Detection - Ready!")


def generate_summary(self):
    """Generate summary dari semua analisis"""
    print("\n" + "="*70)
    print("ANALISIS PCG - RINGKASAN LENGKAP")
    print("="*70)

    summary = {
        'file_path': self.file_path,
        'sampling_rate': self.target_sr,
        'duration': len(self.original_signal) / self.target_sr,
        'steps_completed': []
    }

    print(f"\nFile: {os.path.basename(self.file_path)}")
    print(f"Duration: {summary['duration']:.2f} seconds")
    print(f"Sampling Rate: {self.target_sr} Hz")
    print(f"Total Samples: {len(self.original_signal)}")

    if self.filtered_signal is not None:
        summary['steps_completed'].append('Step 1: Bandpass Filtering')
        print("\n✓ Step 1: Bandpass Filtering - COMPLETED")

    if self.denoised_signal is not None:
        summary['steps_completed'].append('Step 2: Wavelet Denoising')
        print("✓ Step 2: Wavelet Denoising - COMPLETED")

    if self.envelope is not None:
        summary['steps_completed'].append('Step 3: Envelope Extraction')
        print("✓ Step 3: Envelope Extraction - COMPLETED")

    if self.peaks is not None:
        summary['steps_completed'].append('Step 4: Peak Detection')
        print("✓ Step 4: Peak Detection - COMPLETED")
        print(f"  - S1 peaks: {len(self.s1_peaks)}")
        print(f"  - S2 peaks: {len(self.s2_peaks)}")

        if len(self.s1_peaks) > 1:
            heart_rate = 60 / np.mean(np.diff(self.s1_peaks) / self.target_sr)
            print(f"  - Heart Rate: {heart_rate:.1f} BPM")
            summary['heart_rate'] = heart_rate

    print("\n" + "="*70)

    return summary

def save_processed_signals(self, output_dir='/content/outputs'):
    """Simpan semua sinyal yang telah diproses"""
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(self.file_path))[0]

    signals_to_save = {
        'filtered': self.filtered_signal,
        'denoised': self.denoised_signal,
        'envelope': self.envelope,
    }

    saved_files = []
    for name, signal_data in signals_to_save.items():
        if signal_data is not None:
            output_path = os.path.join(output_dir, f"{base_name}_{name}.npy")
            np.save(output_path, signal_data)
            saved_files.append(output_path)

    # Save peaks
    if self.peaks is not None:
        peaks_data = {
            'all_peaks': self.peaks,
            's1_peaks': self.s1_peaks,
            's2_peaks': self.s2_peaks,
            'sampling_rate': self.target_sr
        }
        peaks_path = os.path.join(output_dir, f"{base_name}_peaks.npz")
        np.savez(peaks_path, **peaks_data)
        saved_files.append(peaks_path)

    print(f"\n✓ Processed signals saved to: {output_dir}")
    for file in saved_files:
        print(f"  - {os.path.basename(file)}")

    return saved_files

# Tambahkan method ke class
PCGAnalyzer.generate_summary = generate_summary
PCGAnalyzer.save_processed_signals = save_processed_signals

print("✓ Utility Functions - Ready!")

def analyze_single_pcg(file_path, output_dir='/content/outputs'):
    """
    Fungsi helper untuk menganalisis satu file PCG
    Menjalankan semua 4 step secara otomatis

    Parameters:
    -----------
    file_path : str
        Path ke file PCG (.wav)
    output_dir : str
        Direktori output

    Returns:
    --------
    analyzer : PCGAnalyzer
        Object analyzer dengan semua hasil
    summary : dict
        Summary hasil analisis
    """

    print("PCG ANALYSIS(STEPS 1-4)")

    print(f"\nProcessing: {os.path.basename(file_path)}")
    print("-" * 70)

    try:
        # Inisialisasi
        analyzer = PCGAnalyzer(file_path)

        # Step 1: Bandpass Filtering
        analyzer.bandpass_filter(lowcut=20, highcut=500, order=4, show_plot=True)

        # Step 2: Wavelet Denoising
        analyzer.wavelet_denoise(wavelet='db6', level=6, mode='soft', show_plot=True)

        # Step 3: Envelope Extraction
        analyzer.extract_envelope(show_plot=True)

        # Step 4: Peak Detection
        analyzer.detect_peaks(min_distance=0.3, threshold_factor=0.3, show_plot=True)

        # Generate summary
        summary = analyzer.generate_summary()

        # Save results
        analyzer.save_processed_signals(output_dir)

        print("\n" + "="*70)
        print("✓ ANALISIS SELESAI DENGAN SUKSES!")
        print("="*70)

        return analyzer, summary

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

print("✓ Helper Function - analyze_single_pcg() - Ready!")


def analyze_multiple_pcg(data_dir, output_dir='/content/outputs', max_files=5):
    """
    Fungsi untuk menganalisis multiple files PCG

    Parameters:
    -----------
    data_dir : str
        Direktori dataset
    output_dir : str
        Direktori output
    max_files : int
        Maximum jumlah file yang dianalisis

    Returns:
    --------
    results : list
        List berisi hasil analisis semua files
    """
    print("\n" + "🫀 "*25)
    print("BATCH PCG ANALYSIS - MULTIPLE FILES")
    print("🫀 "*25)

    # Cari semua file .wav
    wav_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))

    wav_files = wav_files[:max_files]

    print(f"\nFound {len(wav_files)} PCG files")
    print("-" * 70)

    results = []
    for i, file_path in enumerate(wav_files, 1):
        print(f"\n[{i}/{len(wav_files)}] Processing: {os.path.basename(file_path)}")

        try:
            analyzer = PCGAnalyzer(file_path)

            # Run all steps tanpa plotting
            analyzer.bandpass_filter(show_plot=False)
            analyzer.wavelet_denoise(show_plot=False)
            analyzer.extract_envelope(show_plot=False)
            analyzer.detect_peaks(show_plot=False)

            summary = analyzer.generate_summary()
            results.append({
                'file': os.path.basename(file_path),
                'analyzer': analyzer,
                'summary': summary
            })

        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    print("\n" + "="*70)
    print(f"✓ BATCH ANALYSIS COMPLETED: {len(results)}/{len(wav_files)} successful")
    print("="*70)

    return results

print("✓ Helper Function - analyze_multiple_pcg() - Ready!")


# Blok 4 - FFT & PSD

def compute_fft(sig, sr):
    print("=== FFT COMPUTATION ===")
    print(f"Signal length (N)         : {len(sig)}")
    print(f"Sampling rate (sr)        : {sr}")

    N = len(sig)
    yf = np.abs(rfft(sig))
    xf = rfftfreq(N, 1/sr)

    print(f"FFT freq bins (len)       : {len(xf)}")
    print(f"FFT magnitude len         : {len(yf)}")
    print(f"Freq range                : {xf[0]:.2f} Hz → {xf[-1]:.2f} Hz")
    print(f"Max FFT magnitude         : {np.max(yf):.6f}")
    
    # show a short preview
    print("First 10 FFT values:")
    for i in range(min(10, len(xf))):
        print(f"  f={xf[i]:8.2f} Hz | mag={yf[i]:.6f}")

    print("")  # line break
    return xf, yf


def compute_psd(sig, sr, nperseg=1024):
    print("=== PSD (WELCH METHOD) ===")
    print(f"Signal length             : {len(sig)}")
    print(f"Sampling rate (sr)        : {sr}")
    print(f"nperseg (segment size)    : {nperseg}")

    f, Pxx = welch(sig, fs=sr, nperseg=nperseg)

    print(f"PSD frequency bins (len)  : {len(f)}")
    print(f"Freq range                : {f[0]:.2f} Hz → {f[-1]:.2f} Hz")
    print(f"Max PSD value             : {np.max(Pxx):.6e}")

    print("First 10 PSD values:")
    for i in range(min(10, len(f))):
        print(f"  f={f[i]:8.2f} Hz | PSD={Pxx[i]:.6e}")

    print("")  # line break
    return f, Pxx


# Blok 5 - spectrogram & scalogram plotting functions

def plot_spectrogram(sig, sr, title="Spectrogram (STFT)"):
    print("=== STFT (SPECTROGRAM) ===")
    print(f"Signal length             : {len(sig)}")
    print(f"Sampling rate (sr)        : {sr}")
    print(f"n_fft                     : 2048")
    print(f"hop_length                : 256")

    D = librosa.stft(sig, n_fft=2048, hop_length=256)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    print(f"STFT matrix shape         : {D.shape}  (freq_bins x time_frames)")
    print(f"Spectrogram dB min/max    : {S_db.min():.2f} dB / {S_db.max():.2f} dB")
    print(f"Number of frequency bins  : {D.shape[0]}")
    print(f"Number of time frames     : {D.shape[1]}\n")

    plt.figure(figsize=(10,4))
    librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_scalogram(sig, sr, wavelet='morl', max_scale=128):
    print("=== CWT SCALOGRAM ===")
    print(f"Signal length             : {len(sig)}")
    print(f"Sampling rate (sr)        : {sr}")
    print(f"Wavelet used              : {wavelet}")
    print(f"Max scale                 : {max_scale}")

    scales = np.arange(1, max_scale)
    print(f"Total scales used         : {len(scales)} (1 → {max_scale-1})")

    coefs, freqs = pywt.cwt(sig, scales, wavelet, 1.0/sr)

    print(f"CWT coefficient shape     : {coefs.shape}  (scales x samples)")
    print(f"Frequency array length    : {len(freqs)}")
    print(f"Freq range                : {freqs.min():.4f} Hz → {freqs.max():.4f} Hz")
    print(f"CWT coeff | min/max/mean  : {np.min(coefs):.5f} / {np.max(coefs):.5f} / {np.mean(coefs):.5f}\n")

    plt.figure(figsize=(10,4))
    plt.imshow(
        np.abs(coefs),
        extent=[0, len(sig)/sr, freqs[-1], freqs[0]], 
        cmap='viridis', 
        aspect='auto'
    )
    plt.yscale('linear')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title("Scalogram (CWT)")
    plt.gca().invert_yaxis()
    plt.colorbar(label='|CWT coeff|')
    plt.show()




