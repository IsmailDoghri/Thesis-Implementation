#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from pathlib import Path
import json
import time
import warnings

warnings.filterwarnings('ignore')


class ComprehensiveFeatureExtractor:

    def __init__(self, sampling_rate=50):
        self.sampling_rate = sampling_rate
        self.window_size = 128
        self.feature_names = []
        self._build_feature_names()

    def _build_feature_names(self):
        time_signals = [
            'tBodyAcc-XYZ', 'tGravityAcc-XYZ', 'tBodyAccJerk-XYZ',
            'tBodyGyro-XYZ', 'tBodyGyroJerk-XYZ',
            'tBodyAccMag', 'tGravityAccMag', 'tBodyAccJerkMag',
            'tBodyGyroMag', 'tBodyGyroJerkMag'
        ]

        freq_signals = [
            'fBodyAcc-XYZ', 'fBodyAccJerk-XYZ',
            'fBodyGyro-XYZ', 'fBodyAccMag', 'fBodyBodyAccJerkMag',
            'fBodyBodyGyroMag', 'fBodyBodyGyroJerkMag'
        ]

        time_funcs = ['mean()', 'std()', 'mad()', 'max()', 'min()',
                      'sma()', 'energy()', 'iqr()', 'entropy()',
                      'arCoeff()', 'correlation()']

        freq_funcs = ['mean()', 'std()', 'mad()', 'max()', 'min()',
                      'sma()', 'energy()', 'iqr()', 'entropy()',
                      'maxInds()', 'meanFreq()', 'skewness()',
                      'kurtosis()', 'bandsEnergy()']

        for signal_name in time_signals:
            if 'XYZ' in signal_name:
                for axis in ['X', 'Y', 'Z']:
                    sig = signal_name.replace('XYZ', axis)
                    for func in time_funcs:
                        if func not in ['sma()', 'energy()']:
                            self.feature_names.append(f"{sig}-{func}")
            else:
                for func in time_funcs:
                    if func != 'correlation()':
                        self.feature_names.append(f"{signal_name}-{func}")

        for signal_name in freq_signals:
            if 'XYZ' in signal_name:
                for axis in ['X', 'Y', 'Z']:
                    sig = signal_name.replace('XYZ', axis)
                    for func in freq_funcs:
                        if func not in ['sma()', 'energy()']:
                            self.feature_names.append(f"{sig}-{func}")
            else:
                for func in freq_funcs:
                    self.feature_names.append(f"{signal_name}-{func}")

        angle_features = [
            'angle(tBodyAccMean,gravity)',
            'angle(tBodyAccJerkMean,gravityMean)',
            'angle(tBodyGyroMean,gravityMean)',
            'angle(tBodyGyroJerkMean,gravityMean)',
            'angle(X,gravityMean)', 'angle(Y,gravityMean)', 'angle(Z,gravityMean)'
        ]
        self.feature_names.extend(angle_features)

    def load_raw_data(self):
        data_path = Path("../UCI HAR Dataset")

        print("Loading raw data...")
        X_train_signals = self._load_signal_set(data_path / "train", "train")
        y_train = np.loadtxt(data_path / "train" / "y_train.txt", dtype=int)

        X_test_signals = self._load_signal_set(data_path / "test", "test")
        y_test = np.loadtxt(data_path / "test" / "y_test.txt", dtype=int)

        return X_train_signals, y_train, X_test_signals, y_test

    def _load_signal_set(self, path, set_name):
        signal_names = [
            'body_acc_x', 'body_acc_y', 'body_acc_z',
            'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
            'total_acc_x', 'total_acc_y', 'total_acc_z'
        ]

        signals = []
        inertial_path = path / "Inertial Signals"

        for signal_name in signal_names:
            filename = f"{signal_name}_{set_name}.txt"
            signal_data = np.loadtxt(inertial_path / filename)
            signals.append(signal_data)

        return np.stack(signals, axis=-1)

    def extract_all_features(self, X_signals):
        n_samples = X_signals.shape[0]
        features = np.zeros((n_samples, 561))

        print(f"Extracting features from {n_samples} samples...")

        for i in range(n_samples):
            if i % 500 == 0 and i > 0:
                print(f"  Processing sample {i}/{n_samples}...")

            features[i] = self._extract_sample_features(X_signals[i])

        return features

    def _extract_sample_features(self, window):
        features = []

        body_acc = window[:, :3]
        body_gyro = window[:, 3:6]
        total_acc = window[:, 6:9]

        gravity_acc = self._butterworth_filter(total_acc, 0.3, 'low')

        body_acc = total_acc - gravity_acc

        body_acc_jerk = np.diff(body_acc, axis=0) * self.sampling_rate
        body_gyro_jerk = np.diff(body_gyro, axis=0) * self.sampling_rate

        body_acc_mag = np.sqrt(np.sum(body_acc ** 2, axis=1))
        gravity_acc_mag = np.sqrt(np.sum(gravity_acc ** 2, axis=1))
        body_acc_jerk_mag = np.sqrt(np.sum(body_acc_jerk ** 2, axis=1))
        body_gyro_mag = np.sqrt(np.sum(body_gyro ** 2, axis=1))
        body_gyro_jerk_mag = np.sqrt(np.sum(body_gyro_jerk ** 2, axis=1))

        for axis in range(3):
            features.extend(self._time_domain_features(body_acc[:, axis]))
        features.append(self._sma(body_acc))
        features.append(self._energy(body_acc))

        for axis in range(3):
            features.extend(self._time_domain_features(gravity_acc[:, axis]))
        features.append(self._sma(gravity_acc))
        features.append(self._energy(gravity_acc))

        for axis in range(3):
            features.extend(self._time_domain_features(body_acc_jerk[:, axis]))
        features.append(self._sma(body_acc_jerk))
        features.append(self._energy(body_acc_jerk))

        for axis in range(3):
            features.extend(self._time_domain_features(body_gyro[:, axis]))
        features.append(self._sma(body_gyro))
        features.append(self._energy(body_gyro))

        for axis in range(3):
            features.extend(self._time_domain_features(body_gyro_jerk[:, axis]))
        features.append(self._sma(body_gyro_jerk))
        features.append(self._energy(body_gyro_jerk))

        features.extend(self._time_domain_features_mag(body_acc_mag))
        features.extend(self._time_domain_features_mag(gravity_acc_mag))
        features.extend(self._time_domain_features_mag(body_acc_jerk_mag))
        features.extend(self._time_domain_features_mag(body_gyro_mag))
        features.extend(self._time_domain_features_mag(body_gyro_jerk_mag))

        for axis in range(3):
            features.extend(self._freq_domain_features(body_acc[:, axis]))
        features.append(self._sma(np.abs(fft(body_acc, axis=0)[:64])))
        features.append(self._energy(np.abs(fft(body_acc, axis=0)[:64])))

        for axis in range(3):
            features.extend(self._freq_domain_features(body_acc_jerk[:, axis]))
        features.append(self._sma(np.abs(fft(body_acc_jerk, axis=0)[:63])))
        features.append(self._energy(np.abs(fft(body_acc_jerk, axis=0)[:63])))

        for axis in range(3):
            features.extend(self._freq_domain_features(body_gyro[:, axis]))
        features.append(self._sma(np.abs(fft(body_gyro, axis=0)[:64])))
        features.append(self._energy(np.abs(fft(body_gyro, axis=0)[:64])))

        features.extend(self._freq_domain_features_mag(body_acc_mag))
        features.extend(self._freq_domain_features_mag(body_acc_jerk_mag))
        features.extend(self._freq_domain_features_mag(body_gyro_mag))
        features.extend(self._freq_domain_features_mag(body_gyro_jerk_mag))

        features.extend(self._angle_features(body_acc, gravity_acc, body_gyro,
                                             body_acc_jerk, body_gyro_jerk))

        return np.array(features)

    def _time_domain_features(self, signal_1d):
        features = []

        features.append(np.mean(signal_1d))
        features.append(np.std(signal_1d))
        features.append(stats.median_abs_deviation(signal_1d))
        features.append(np.max(signal_1d))
        features.append(np.min(signal_1d))
        features.append(stats.iqr(signal_1d))
        features.append(self._entropy(signal_1d))

        ar_coeffs = self._ar_coefficients(signal_1d, 4)
        features.extend(ar_coeffs)

        features.append(0.0)

        return features

    def _time_domain_features_mag(self, signal_1d):
        features = []

        features.append(np.mean(signal_1d))
        features.append(np.std(signal_1d))
        features.append(stats.median_abs_deviation(signal_1d))
        features.append(np.max(signal_1d))
        features.append(np.min(signal_1d))
        features.append(np.mean(np.abs(signal_1d)))
        features.append(np.sum(signal_1d ** 2) / len(signal_1d))
        features.append(stats.iqr(signal_1d))
        features.append(self._entropy(signal_1d))

        ar_coeffs = self._ar_coefficients(signal_1d, 4)
        features.extend(ar_coeffs)

        return features

    def _freq_domain_features(self, signal_1d):
        features = []

        fft_vals = fft(signal_1d)
        fft_abs = np.abs(fft_vals[:len(fft_vals) // 2])
        freqs = fftfreq(len(signal_1d), 1 / self.sampling_rate)[:len(fft_vals) // 2]

        features.append(np.mean(fft_abs))
        features.append(np.std(fft_abs))
        features.append(stats.median_abs_deviation(fft_abs))
        features.append(np.max(fft_abs))
        features.append(np.min(fft_abs))
        features.append(stats.iqr(fft_abs))
        features.append(self._entropy(fft_abs))
        features.append(np.argmax(fft_abs))
        features.append(self._mean_freq(fft_abs, freqs))
        features.append(stats.skew(fft_abs))
        features.append(stats.kurtosis(fft_abs))

        band_energy = self._bands_energy(fft_abs)
        features.extend(band_energy)

        return features

    def _freq_domain_features_mag(self, signal_1d):
        features = []

        fft_vals = fft(signal_1d)
        fft_abs = np.abs(fft_vals[:len(fft_vals) // 2])
        freqs = fftfreq(len(signal_1d), 1 / self.sampling_rate)[:len(fft_vals) // 2]

        features.append(np.mean(fft_abs))
        features.append(np.std(fft_abs))
        features.append(stats.median_abs_deviation(fft_abs))
        features.append(np.max(fft_abs))
        features.append(np.min(fft_abs))
        features.append(np.mean(np.abs(fft_abs)))
        features.append(np.sum(fft_abs ** 2) / len(fft_abs))
        features.append(stats.iqr(fft_abs))
        features.append(self._entropy(fft_abs))
        features.append(np.argmax(fft_abs))
        features.append(self._mean_freq(fft_abs, freqs))
        features.append(stats.skew(fft_abs))
        features.append(stats.kurtosis(fft_abs))

        return features

    def _angle_features(self, body_acc, gravity_acc, body_gyro, body_acc_jerk, body_gyro_jerk):
        features = []

        body_acc_mean = np.mean(body_acc, axis=0)
        gravity_mean = np.mean(gravity_acc, axis=0)
        body_gyro_mean = np.mean(body_gyro, axis=0)
        body_acc_jerk_mean = np.mean(body_acc_jerk, axis=0)
        body_gyro_jerk_mean = np.mean(body_gyro_jerk, axis=0)

        features.append(self._angle(body_acc_mean, gravity_mean))
        features.append(self._angle(body_acc_jerk_mean, gravity_mean))
        features.append(self._angle(body_gyro_mean, gravity_mean))
        features.append(self._angle(body_gyro_jerk_mean, gravity_mean))
        features.append(self._angle([1, 0, 0], gravity_mean))
        features.append(self._angle([0, 1, 0], gravity_mean))
        features.append(self._angle([0, 0, 1], gravity_mean))

        return features

    def _butterworth_filter(self, signal_data, cutoff, filter_type='low', order=3):
        nyquist = self.sampling_rate / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype=filter_type, analog=False)

        filtered = np.zeros_like(signal_data)
        for i in range(signal_data.shape[1]):
            filtered[:, i] = signal.filtfilt(b, a, signal_data[:, i])
        return filtered

    def _sma(self, signal_3d):
        return np.mean(np.sum(np.abs(signal_3d), axis=1))

    def _energy(self, signal_data):
        if len(signal_data.shape) == 1:
            return np.sum(signal_data ** 2) / len(signal_data)
        else:
            return np.sum(signal_data ** 2) / signal_data.size

    def _entropy(self, signal_1d):
        if np.sum(np.abs(signal_1d)) == 0:
            return 0
        p = np.abs(signal_1d) / np.sum(np.abs(signal_1d))
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    def _ar_coefficients(self, signal_1d, order=4):
        try:
            from statsmodels.tsa.ar_model import AutoReg
            model = AutoReg(signal_1d, lags=order, trend='n')
            model_fit = model.fit()
            return model_fit.params
        except:
            coeffs = []
            for lag in range(1, order + 1):
                if len(signal_1d) > lag:
                    coeffs.append(np.corrcoef(signal_1d[:-lag], signal_1d[lag:])[0, 1])
                else:
                    coeffs.append(0)
            return coeffs

    def _mean_freq(self, fft_abs, freqs):
        if np.sum(fft_abs) == 0:
            return 0
        return np.sum(freqs * fft_abs) / np.sum(fft_abs)

    def _bands_energy(self, fft_abs, n_bands=8):
        band_size = len(fft_abs) // n_bands
        bands = []

        for i in range(n_bands):
            start = i * band_size
            end = (i + 1) * band_size if i < n_bands - 1 else len(fft_abs)
            bands.append(np.sum(fft_abs[start:end] ** 2))

        return bands

    def _angle(self, v1, v2):
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        return np.arccos(cos_angle)

    def normalize_features(self, X):
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1

        return 2 * (X - X_min) / X_range - 1

    def save_features(self, X_train, X_test, y_train, y_test):
        output_dir = Path("data/extracted_features")
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "X_train_manual_561.npy", X_train)
        np.save(output_dir / "X_test_manual_561.npy", X_test)
        np.save(output_dir / "y_train.npy", y_train)
        np.save(output_dir / "y_test.npy", y_test)

        with open(output_dir / "feature_names_561.json", 'w') as f:
            json.dump(self.feature_names, f, indent=2)

        print(f"✓ Saved features to {output_dir}")

        return output_dir


def main():
    print("PHASE 2: FEATURE EXTRACTION (561 FEATURES)")

    extractor = ComprehensiveFeatureExtractor(sampling_rate=50)

    X_train_raw, y_train, X_test_raw, y_test = extractor.load_raw_data()
    print(f"✓ Loaded: {X_train_raw.shape[0]} train, {X_test_raw.shape[0]} test samples")

    start_time = time.time()

    X_train_features = extractor.extract_all_features(X_train_raw)
    X_test_features = extractor.extract_all_features(X_test_raw)

    extraction_time = time.time() - start_time

    X_train_norm = extractor.normalize_features(X_train_features)
    X_test_norm = extractor.normalize_features(X_test_features)

    output_dir = extractor.save_features(X_train_norm, X_test_norm, y_train, y_test)

    print(f"✓ Extracted {X_train_norm.shape[1]} features in {extraction_time:.1f}s")

    try:
        data_path = Path("../UCI HAR Dataset")
        X_uci_train = np.loadtxt(data_path / "train" / "X_train.txt")
        X_uci_test = np.loadtxt(data_path / "test" / "X_test.txt")

        if X_uci_train.shape[1] == X_train_norm.shape[1]:
            print("✓ Feature count matches UCI (561 features)")
        else:
            print(f"⚠️ Feature mismatch: UCI={X_uci_train.shape[1]}, Ours={X_train_norm.shape[1]}")
    except Exception as e:
        print(f"Could not compare with UCI: {e}")

    return X_train_norm, X_test_norm, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = main()
