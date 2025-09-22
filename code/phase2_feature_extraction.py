import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import os
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings('ignore')


class FeatureExtractor:
    def __init__(self):
        self.sampling_freq = 50
        self.window_size = 128
        self.overlap = 64

    def extract_time_domain_features(self, signal_window):
        features = {}

        features['mean'] = np.mean(signal_window)
        features['std'] = np.std(signal_window)
        features['mad'] = np.mean(np.abs(signal_window - np.mean(signal_window)))
        features['max'] = np.max(signal_window)
        features['min'] = np.min(signal_window)
        features['sma'] = np.sum(np.abs(signal_window)) / len(signal_window)
        features['energy'] = np.sum(signal_window ** 2) / len(signal_window)
        features['iqr'] = np.percentile(signal_window, 75) - np.percentile(signal_window, 25)

        features['entropy'] = -np.sum(np.histogram(signal_window, bins=10)[0] *
                                      np.log(np.histogram(signal_window, bins=10)[0] + 1e-10))
        features['skewness'] = stats.skew(signal_window)
        features['kurtosis'] = stats.kurtosis(signal_window)

        try:
            from statsmodels.tsa.ar_model import AutoReg
            ar_model = AutoReg(signal_window, lags=4, old_names=False)
            ar_result = ar_model.fit()
            ar_coeffs = ar_result.params[1:]
            for i, coeff in enumerate(ar_coeffs):
                features[f'ar{i + 1}'] = coeff
        except:
            for i in range(4):
                features[f'ar{i + 1}'] = 0.0

        return features

    def extract_frequency_domain_features(self, signal_window):
        features = {}

        n = len(signal_window)
        fft_vals = fft(signal_window)
        fft_freqs = fftfreq(n, d=1 / self.sampling_freq)

        positive_freqs = fft_freqs[:n // 2]
        positive_fft = np.abs(fft_vals[:n // 2])

        features['mean_freq'] = np.mean(positive_fft)
        features['std_freq'] = np.std(positive_fft)
        features['mad_freq'] = np.mean(np.abs(positive_fft - np.mean(positive_fft)))
        features['max_freq'] = np.max(positive_fft)
        features['min_freq'] = np.min(positive_fft)
        features['sma_freq'] = np.sum(positive_fft) / len(positive_fft)
        features['energy_freq'] = np.sum(positive_fft ** 2) / len(positive_fft)

        psd = positive_fft ** 2
        psd_norm = psd / np.sum(psd)
        features['entropy_freq'] = -np.sum(psd_norm * np.log(psd_norm + 1e-10))

        max_idx = np.argmax(positive_fft)
        features['max_inds'] = positive_freqs[max_idx] if max_idx < len(positive_freqs) else 0
        features['mean_freq_weighted'] = np.sum(positive_freqs * positive_fft) / np.sum(positive_fft)

        features['skewness_freq'] = stats.skew(positive_fft)
        features['kurtosis_freq'] = stats.kurtosis(positive_fft)

        walking_band = (positive_freqs >= 1) & (positive_freqs <= 3)
        features['energy_walking_band'] = np.sum(positive_fft[walking_band] ** 2) if np.any(walking_band) else 0

        postural_band = (positive_freqs >= 0.1) & (positive_freqs <= 0.5)
        features['energy_postural_band'] = np.sum(positive_fft[postural_band] ** 2) if np.any(postural_band) else 0

        n_bands = 8
        band_width = len(positive_fft) // n_bands
        for i in range(n_bands):
            start_idx = i * band_width
            end_idx = min((i + 1) * band_width, len(positive_fft))
            features[f'bands_energy_{i + 1}'] = np.sum(positive_fft[start_idx:end_idx] ** 2)

        return features

    def extract_angle_features(self, accel_data):
        features = {}

        mean_x = np.mean(accel_data['x'])
        mean_y = np.mean(accel_data['y'])
        mean_z = np.mean(accel_data['z'])

        gravity = np.array([mean_x, mean_y, mean_z])
        gravity_norm = np.linalg.norm(gravity)

        if gravity_norm > 0:
            gravity = gravity / gravity_norm

            features['angle_x_gravity'] = np.arccos(np.clip(gravity[0], -1, 1))
            features['angle_y_gravity'] = np.arccos(np.clip(gravity[1], -1, 1))
            features['angle_z_gravity'] = np.arccos(np.clip(gravity[2], -1, 1))
        else:
            features['angle_x_gravity'] = 0
            features['angle_y_gravity'] = 0
            features['angle_z_gravity'] = 0

        return features

    def extract_magnitude_features(self, x, y, z):
        return np.sqrt(x ** 2 + y ** 2 + z ** 2)

    def extract_jerk_signal(self, signal_data, delta_t=1 / 50):
        return np.diff(signal_data) / delta_t

    def extract_all_features_from_window(self, raw_signals):
        all_features = []

        signal_types = {
            'body_acc': raw_signals.get('body_acc', None),
            'body_gyro': raw_signals.get('body_gyro', None),
            'total_acc': raw_signals.get('total_acc', None)
        }

        for signal_name, signal_data in signal_types.items():
            if signal_data is None:
                continue

            for axis in range(3):
                axis_signal = signal_data[:, axis]

                time_features = self.extract_time_domain_features(axis_signal)
                all_features.extend(time_features.values())

                freq_features = self.extract_frequency_domain_features(axis_signal)
                all_features.extend(freq_features.values())

                if 'acc' in signal_name:
                    jerk_signal = self.extract_jerk_signal(axis_signal)
                    jerk_time_features = self.extract_time_domain_features(jerk_signal)
                    all_features.extend(jerk_time_features.values())
                    jerk_freq_features = self.extract_frequency_domain_features(jerk_signal)
                    all_features.extend(jerk_freq_features.values())

            if signal_data.shape[1] == 3:
                magnitude = self.extract_magnitude_features(
                    signal_data[:, 0], signal_data[:, 1], signal_data[:, 2]
                )
                mag_time_features = self.extract_time_domain_features(magnitude)
                all_features.extend(mag_time_features.values())
                mag_freq_features = self.extract_frequency_domain_features(magnitude)
                all_features.extend(mag_freq_features.values())

                if 'acc' in signal_name:
                    jerk_mag = self.extract_jerk_signal(magnitude)
                    jerk_mag_time = self.extract_time_domain_features(jerk_mag)
                    all_features.extend(jerk_mag_time.values())
                    jerk_mag_freq = self.extract_frequency_domain_features(jerk_mag)
                    all_features.extend(jerk_mag_freq.values())

        if 'body_acc' in signal_types and signal_types['body_acc'] is not None:
            angle_features = self.extract_angle_features({
                'x': signal_types['body_acc'][:, 0],
                'y': signal_types['body_acc'][:, 1],
                'z': signal_types['body_acc'][:, 2]
            })
            all_features.extend(angle_features.values())

        return np.array(all_features)


def load_raw_inertial_signals(base_path, dataset='train'):
    signals = {}
    signal_types = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z'
    ]

    inertial_path = os.path.join(base_path, dataset, f'Inertial Signals')

    for signal_type in signal_types:
        file_path = os.path.join(inertial_path, f'{signal_type}_{dataset}.txt')
        if os.path.exists(file_path):
            signals[signal_type] = np.loadtxt(file_path)

    organized_signals = {}

    if 'body_acc_x' in signals:
        organized_signals['body_acc'] = np.stack([
            signals['body_acc_x'],
            signals['body_acc_y'],
            signals['body_acc_z']
        ], axis=2)

    if 'body_gyro_x' in signals:
        organized_signals['body_gyro'] = np.stack([
            signals['body_gyro_x'],
            signals['body_gyro_y'],
            signals['body_gyro_z']
        ], axis=2)

    if 'total_acc_x' in signals:
        organized_signals['total_acc'] = np.stack([
            signals['total_acc_x'],
            signals['total_acc_y'],
            signals['total_acc_z']
        ], axis=2)

    return organized_signals


def main():
    print("PHASE 2: FEATURE EXTRACTION")

    results_dir = 'results/phase2'
    os.makedirs(results_dir, exist_ok=True)

    extractor = FeatureExtractor()

    base_path = '/Users/ismaildoghri/Downloads/MSc Dissertation /Chosen Datasets/UCI-HAR/UCI HAR Dataset'

    print("Loading data...")
    train_signals = load_raw_inertial_signals(base_path, 'train')
    test_signals = load_raw_inertial_signals(base_path, 'test')

    y_train = np.loadtxt(os.path.join(base_path, 'train', 'y_train.txt'))
    y_test = np.loadtxt(os.path.join(base_path, 'test', 'y_test.txt'))

    print(f"✓ Loaded: {train_signals['body_acc'].shape[0]} train, {test_signals['body_acc'].shape[0]} test samples")

    n_samples_to_process = 100

    extracted_features_train = []
    extracted_features_test = []

    print("Extracting features...")
    for i in range(min(n_samples_to_process, len(train_signals['body_acc']))):
        if i % 20 == 0 and i > 0:
            print(f"  Processing sample {i}/{n_samples_to_process}")

        window_data = {
            'body_acc': train_signals['body_acc'][i],
            'body_gyro': train_signals['body_gyro'][i],
            'total_acc': train_signals['total_acc'][i]
        }
        features = extractor.extract_all_features_from_window(window_data)
        extracted_features_train.append(features)

    extracted_features_train = np.array(extracted_features_train)

    for i in range(min(n_samples_to_process // 2, len(test_signals['body_acc']))):
        window_data = {
            'body_acc': test_signals['body_acc'][i],
            'body_gyro': test_signals['body_gyro'][i],
            'total_acc': test_signals['total_acc'][i]
        }
        features = extractor.extract_all_features_from_window(window_data)
        extracted_features_test.append(features)

    extracted_features_test = np.array(extracted_features_test)
    print(f"✓ Extracted: {extracted_features_train.shape[1]} features per sample")

    X_train_uci = np.loadtxt(os.path.join(base_path, 'train', 'X_train.txt'))
    X_test_uci = np.loadtxt(os.path.join(base_path, 'test', 'X_test.txt'))

    print(f"✓ UCI features: {X_train_uci.shape[1]} features")

    feature_names_path = os.path.join(base_path, 'features.txt')
    feature_names = []
    with open(feature_names_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                feature_names.append(' '.join(parts[1:]))

    time_features = [f for f in feature_names if not 'Freq' in f and not 'freq' in f]
    freq_features = [f for f in feature_names if 'Freq' in f or 'freq' in f]

    feature_types = {
        'mean': [f for f in feature_names if 'mean()' in f.lower()],
        'std': [f for f in feature_names if 'std()' in f.lower()],
        'mad': [f for f in feature_names if 'mad()' in f.lower()],
        'max': [f for f in feature_names if 'max()' in f.lower()],
        'min': [f for f in feature_names if 'min()' in f.lower()],
        'sma': [f for f in feature_names if 'sma()' in f.lower()],
        'energy': [f for f in feature_names if 'energy()' in f.lower()],
        'iqr': [f for f in feature_names if 'iqr()' in f.lower()],
        'entropy': [f for f in feature_names if 'entropy()' in f.lower()],
        'ar': [f for f in feature_names if 'arCoeff()' in f.lower()],
        'correlation': [f for f in feature_names if 'correlation()' in f.lower()],
        'bands': [f for f in feature_names if 'bandsEnergy()' in f.lower()]
    }

    signal_sources = {
        'BodyAcc': [f for f in feature_names if 'BodyAcc' in f],
        'GravityAcc': [f for f in feature_names if 'GravityAcc' in f],
        'BodyGyro': [f for f in feature_names if 'BodyGyro' in f],
        'BodyAccJerk': [f for f in feature_names if 'BodyAccJerk' in f],
        'BodyGyroJerk': [f for f in feature_names if 'BodyGyroJerk' in f]
    }

    print("Normalizing features...")
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_normalized = scaler.fit_transform(X_train_uci)
    X_test_normalized = scaler.transform(X_test_uci)

    test_outliers = np.sum((X_test_normalized < -1) | (X_test_normalized > 1))

    if test_outliers > 0:
        print(f"  ⚠️ Test set has {test_outliers} outliers")
        X_test_normalized_clipped = np.clip(X_test_normalized, -1, 1)

    activity_names = {
        1: 'WALKING',
        2: 'WALKING_UPSTAIRS',
        3: 'WALKING_DOWNSTAIRS',
        4: 'SITTING',
        5: 'STANDING',
        6: 'LAYING'
    }

    for activity_id, activity_name in activity_names.items():
        activity_mask = y_train == activity_id
        activity_features = X_train_normalized[activity_mask]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Phase 2: Feature Extraction Analysis', fontsize=16, fontweight='bold')

    ax = axes[0, 0]
    feat_type_counts = [len(v) for v in feature_types.values()]
    feat_type_names = list(feature_types.keys())
    ax.bar(feat_type_names, feat_type_counts, color='steelblue')
    ax.set_xlabel('Feature Type')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Feature Types')
    ax.tick_params(axis='x', rotation=45)

    ax = axes[0, 1]
    domain_counts = [len(time_features), len(freq_features)]
    domain_names = ['Time Domain', 'Frequency Domain']
    colors = ['forestgreen', 'coral']
    ax.pie(domain_counts, labels=domain_names, colors=colors, autopct='%1.1f%%')
    ax.set_title('Time vs Frequency Domain Features')

    ax = axes[0, 2]
    source_counts = [len(v) for k, v in signal_sources.items()]
    source_names = list(signal_sources.keys())
    ax.barh(source_names, source_counts, color='teal')
    ax.set_xlabel('Number of Features')
    ax.set_title('Features by Signal Source')

    ax = axes[1, 0]
    ax.hist(X_train_normalized.flatten(), bins=50, alpha=0.7, label='Train', color='blue', density=True)
    ax.hist(X_test_normalized.flatten(), bins=50, alpha=0.7, label='Test', color='red', density=True)
    ax.set_xlabel('Normalized Feature Value')
    ax.set_ylabel('Density')
    ax.set_title('Feature Value Distribution')
    ax.legend()
    ax.axvline(-1, color='black', linestyle='--', alpha=0.5)
    ax.axvline(1, color='black', linestyle='--', alpha=0.5)

    ax = axes[1, 1]
    activity_means = []
    activity_stds = []
    for activity_id in range(1, 7):
        activity_mask = y_train == activity_id
        activity_features = X_train_normalized[activity_mask]
        activity_means.append(np.mean(activity_features))
        activity_stds.append(np.std(activity_features))

    x_pos = np.arange(len(activity_names))
    ax.bar(x_pos, activity_means, yerr=activity_stds, capsize=5, color='purple', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([activity_names[i + 1] for i in range(6)], rotation=45, ha='right')
    ax.set_ylabel('Mean Feature Value')
    ax.set_title('Average Feature Values by Activity')
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)

    ax = axes[1, 2]
    if extracted_features_train.shape[0] > 0:
        sample_features = extracted_features_train[0, :50]
        ax.plot(sample_features, 'g-', alpha=0.7, linewidth=1)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Value')
        ax.set_title('Sample Extracted Features (First 50)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'phase2_feature_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization: phase2_feature_analysis.png")

    summary_content = f"""
PHASE 2: FEATURE EXTRACTION SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

Dataset: {len(train_signals['body_acc'])} train, {len(test_signals['body_acc'])} test samples
UCI Features: {len(feature_names)} total ({len(time_features)} time, {len(freq_features)} frequency)
Our Extraction: {extracted_features_train.shape[1]} features
Normalization: [-1, 1] range with {test_outliers} outliers

Activities:
"""

    for activity_id, activity_name in activity_names.items():
        activity_mask = y_train == activity_id
        activity_features = X_train_normalized[activity_mask]

        summary_content += f"  {activity_name}: {np.sum(activity_mask)} samples\n"

    summary_content += f"""
================================================================================
Phase 2 Completed Successfully
"""

    summary_path = os.path.join(results_dir, 'phase2_comprehensive_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_content)

    results_json = {
        'dataset': {
            'n_train': len(train_signals['body_acc']),
            'n_test': len(test_signals['body_acc']),
            'n_features_uci': len(feature_names),
            'n_features_extracted': int(extracted_features_train.shape[1]),
            'sampling_freq_hz': 50,
            'window_size_samples': 128,
            'window_overlap_samples': 64
        },
        'feature_distribution': {
            'time_domain': len(time_features),
            'frequency_domain': len(freq_features),
            'time_percentage': round(100 * len(time_features) / len(feature_names), 2),
            'freq_percentage': round(100 * len(freq_features) / len(feature_names), 2)
        },
        'feature_types': {k: len(v) for k, v in feature_types.items()},
        'signal_sources': {k: len(v) for k, v in signal_sources.items()},
        'normalization': {
            'train_original_range': [float(np.min(X_train_uci)), float(np.max(X_train_uci))],
            'test_original_range': [float(np.min(X_test_uci)), float(np.max(X_test_uci))],
            'train_normalized_range': [float(np.min(X_train_normalized)), float(np.max(X_train_normalized))],
            'test_normalized_range': [float(np.min(X_test_normalized)), float(np.max(X_test_normalized))],
            'test_outliers': int(test_outliers),
            'test_outlier_percentage': round(
                100 * test_outliers / (X_test_normalized.shape[0] * X_test_normalized.shape[1]), 4)
        },
        'activity_stats': {}
    }

    for activity_id, activity_name in activity_names.items():
        activity_mask = y_train == activity_id
        activity_features = X_train_normalized[activity_mask]
        results_json['activity_stats'][activity_name] = {
            'n_samples': int(np.sum(activity_mask)),
            'percentage': round(100 * np.sum(activity_mask) / len(y_train), 2),
            'mean': float(np.mean(activity_features)),
            'std': float(np.std(activity_features)),
            'min': float(np.min(activity_features)),
            'max': float(np.max(activity_features)),
            'median': float(np.median(activity_features))
        }

    json_path = os.path.join(results_dir, 'phase2_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    np.save(os.path.join(results_dir, 'X_train_normalized.npy'), X_train_normalized)
    np.save(os.path.join(results_dir, 'X_test_normalized.npy'), X_test_normalized)
    if test_outliers > 0:
        np.save(os.path.join(results_dir, 'X_test_normalized_clipped.npy'), X_test_normalized_clipped)

    print(f"✓ Saved files to {results_dir}")
    print("✓ PHASE 2 COMPLETED")


if __name__ == '__main__':
    main()
