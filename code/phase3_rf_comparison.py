#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from scipy.fft import fft, fftfreq


class SimpleFeatureExtractor:

    def __init__(self, sampling_rate=50):
        self.sampling_rate = sampling_rate

    def extract_features(self, raw_signals):
        n_samples = raw_signals.shape[0]
        features = []

        for i in range(n_samples):
            if i % 500 == 0 and i > 0:
                print(f"  Processing sample {i}/{n_samples}...")

            sample_features = []
            window = raw_signals[i]

            for ch in range(9):
                signal_1d = window[:, ch]

                sample_features.append(np.mean(signal_1d))
                sample_features.append(np.std(signal_1d))
                sample_features.append(np.max(signal_1d))
                sample_features.append(np.min(signal_1d))
                sample_features.append(stats.median_abs_deviation(signal_1d))
                sample_features.append(stats.iqr(signal_1d))
                sample_features.append(np.sum(signal_1d ** 2) / len(signal_1d))

                fft_vals = np.abs(fft(signal_1d))[:64]
                sample_features.append(np.mean(fft_vals))
                sample_features.append(np.std(fft_vals))
                sample_features.append(np.max(fft_vals))
                sample_features.append(np.argmax(fft_vals))
                sample_features.append(stats.skew(fft_vals))
                sample_features.append(stats.kurtosis(fft_vals))

                jerk = np.diff(signal_1d)
                sample_features.append(np.mean(jerk))
                sample_features.append(np.std(jerk))

            acc_mag = np.sqrt(np.sum(window[:, :3] ** 2, axis=1))
            gyro_mag = np.sqrt(np.sum(window[:, 3:6] ** 2, axis=1))

            sample_features.append(np.mean(acc_mag))
            sample_features.append(np.std(acc_mag))
            sample_features.append(np.max(acc_mag))
            sample_features.append(np.mean(gyro_mag))
            sample_features.append(np.std(gyro_mag))
            sample_features.append(np.max(gyro_mag))

            features.append(sample_features)

        return np.array(features)


class RandomForestComparison:

    def __init__(self):
        self.activities = {
            1: 'WALKING',
            2: 'WALKING_UPSTAIRS',
            3: 'WALKING_DOWNSTAIRS',
            4: 'SITTING',
            5: 'STANDING',
            6: 'LAYING'
        }

    def load_uci_features(self):
        data_path = Path("../UCI HAR Dataset")

        X_train_uci = np.loadtxt(data_path / "train" / "X_train.txt")
        X_test_uci = np.loadtxt(data_path / "test" / "X_test.txt")
        y_train = np.loadtxt(data_path / "train" / "y_train.txt", dtype=int)
        y_test = np.loadtxt(data_path / "test" / "y_test.txt", dtype=int)

        return X_train_uci, X_test_uci, y_train, y_test

    def load_raw_signals(self):
        data_path = Path("../UCI HAR Dataset")

        train_signals = []
        test_signals = []

        signal_names = [
            'body_acc_x', 'body_acc_y', 'body_acc_z',
            'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
            'total_acc_x', 'total_acc_y', 'total_acc_z'
        ]

        for signal_name in signal_names:
            train_file = data_path / "train" / "Inertial Signals" / f"{signal_name}_train.txt"
            test_file = data_path / "test" / "Inertial Signals" / f"{signal_name}_test.txt"
            train_signals.append(np.loadtxt(train_file))
            test_signals.append(np.loadtxt(test_file))

        X_train_raw = np.stack(train_signals, axis=-1)
        X_test_raw = np.stack(test_signals, axis=-1)

        return X_train_raw, X_test_raw

    def train_random_forest(self, X_train, y_train, feature_type="UCI"):
        print(f"Training RF with {feature_type} features ({X_train.shape[1]} features)...")

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )

        start_time = time.time()
        rf.fit(X_train, y_train)
        train_time = time.time() - start_time

        print(f"✓ Training completed in {train_time:.2f}s")
        return rf, train_time

    def evaluate_model(self, model, X_test, y_test, feature_type="UCI"):
        start_time = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start_time

        accuracy = accuracy_score(y_test, y_pred)
        print(f"✓ {feature_type} Accuracy: {accuracy:.4f}")

        report = classification_report(y_test, y_pred, target_names=list(self.activities.values()), output_dict=True)

        return {
            'accuracy': accuracy,
            'train_time': getattr(model, 'train_time', 0),
            'pred_time': pred_time,
            'classification_report': report,
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

    def compare_features(self):
        print("RANDOM FOREST FEATURE COMPARISON")

        results = {}

        print("\n1. UCI Features")
        X_train_uci, X_test_uci, y_train, y_test = self.load_uci_features()
        rf_uci, train_time_uci = self.train_random_forest(X_train_uci, y_train, "UCI")
        rf_uci.train_time = train_time_uci
        results['uci'] = self.evaluate_model(rf_uci, X_test_uci, y_test, "UCI")

        print("\n2. Custom Features")
        X_train_raw, X_test_raw = self.load_raw_signals()

        print("Extracting features...")
        extractor = SimpleFeatureExtractor()

        X_train_custom = extractor.extract_features(X_train_raw)
        X_test_custom = extractor.extract_features(X_test_raw)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_custom = scaler.fit_transform(X_train_custom)
        X_test_custom = scaler.transform(X_test_custom)

        rf_custom, train_time_custom = self.train_random_forest(X_train_custom, y_train, "Custom")
        rf_custom.train_time = train_time_custom
        results['custom'] = self.evaluate_model(rf_custom, X_test_custom, y_test, "Custom")

        uci_importance = rf_uci.feature_importances_
        top_20_uci = np.argsort(uci_importance)[-20:][::-1]

        custom_importance = rf_custom.feature_importances_
        top_20_custom = np.argsort(custom_importance)[-20:][::-1]

        print(f"\nSUMMARY:")
        print(f"UCI: {results['uci']['accuracy']:.4f} accuracy")
        print(f"Custom: {results['custom']['accuracy']:.4f} accuracy")

        acc_diff = results['uci']['accuracy'] - results['custom']['accuracy']
        print(f"Difference: {acc_diff:.4f}")

        if acc_diff > 0:
            print("→ UCI features perform better")
        else:
            print("→ Custom features perform better")

        for activity_name in self.activities.values():
            uci_acc = results['uci']['classification_report'][activity_name]['recall']
            custom_acc = results['custom']['classification_report'][activity_name]['recall']
            diff = uci_acc - custom_acc

        self._save_results(results, X_train_uci.shape[1], X_train_custom.shape[1])

        return results

    def _generate_feature_names(self):
        names = []
        channels = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z',
                    'total_acc_x', 'total_acc_y', 'total_acc_z']

        for ch in channels:
            names.extend([
                f'{ch}_mean', f'{ch}_std', f'{ch}_max', f'{ch}_min',
                f'{ch}_mad', f'{ch}_iqr', f'{ch}_energy',
                f'{ch}_fft_mean', f'{ch}_fft_std', f'{ch}_fft_max',
                f'{ch}_fft_maxfreq', f'{ch}_fft_skew', f'{ch}_fft_kurt',
                f'{ch}_jerk_mean', f'{ch}_jerk_std'
            ])

        names.extend(['acc_mag_mean', 'acc_mag_std', 'acc_mag_max',
                      'gyro_mag_mean', 'gyro_mag_std', 'gyro_mag_max'])

        return names

    def _save_results(self, results, n_uci_features, n_custom_features):
        output = {
            'feature_counts': {
                'uci': n_uci_features,
                'custom': n_custom_features
            },
            'accuracy': {
                'uci': float(results['uci']['accuracy']),
                'custom': float(results['custom']['accuracy']),
                'difference': float(results['uci']['accuracy'] - results['custom']['accuracy'])
            },
            'training_time': {
                'uci': results['uci']['train_time'],
                'custom': results['custom']['train_time']
            },
            'per_activity': {}
        }

        for activity_name in self.activities.values():
            output['per_activity'][activity_name] = {
                'uci': results['uci']['classification_report'][activity_name]['recall'],
                'custom': results['custom']['classification_report'][activity_name]['recall']
            }

        with open('rf_feature_comparison_results.json', 'w') as f:
            json.dump(output, f, indent=2)

        print(f"✓ Saved results to rf_feature_comparison_results.json")


def main():
    comparator = RandomForestComparison()
    results = comparator.compare_features()

    print("✓ ANALYSIS COMPLETE")

    return results


if __name__ == "__main__":
    results = main()
