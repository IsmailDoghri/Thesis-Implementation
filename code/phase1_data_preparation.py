#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
UCI_HAR_PATH = BASE_DIR.parent / 'UCI HAR Dataset'
RESULTS_DIR = BASE_DIR / 'results' / 'phase1'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ACTIVITIES = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING'
}


class Phase1DataPreparation:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.subjects_train = None
        self.subjects_test = None
        self.raw_signals_train = None
        self.raw_signals_test = None

    def load_dataset(self):
        print("PHASE 1.1: DATASET SETUP")

        if not UCI_HAR_PATH.exists():
            print(f"ERROR: UCI HAR Dataset not found at {UCI_HAR_PATH}")
            sys.exit(1)

        print("Loading dataset...")
        self.X_train = np.loadtxt(UCI_HAR_PATH / 'train' / 'X_train.txt')
        self.X_test = np.loadtxt(UCI_HAR_PATH / 'test' / 'X_test.txt')

        self.y_train = np.loadtxt(UCI_HAR_PATH / 'train' / 'y_train.txt', dtype=int)
        self.y_test = np.loadtxt(UCI_HAR_PATH / 'test' / 'y_test.txt', dtype=int)

        self.subjects_train = np.loadtxt(UCI_HAR_PATH / 'train' / 'subject_train.txt', dtype=int)
        self.subjects_test = np.loadtxt(UCI_HAR_PATH / 'test' / 'subject_test.txt', dtype=int)

        n_train_samples = self.X_train.shape[0]
        n_test_samples = self.X_test.shape[0]
        n_features = self.X_train.shape[1]
        n_total = n_train_samples + n_test_samples

        print(f"✓ Loaded: {n_total} total samples, {n_features} features")

        activities = np.unique(np.concatenate([self.y_train, self.y_test]))
        subjects = np.unique(np.concatenate([self.subjects_train, self.subjects_test]))

    def load_raw_signals(self):
        signal_types = [
            'body_acc_x', 'body_acc_y', 'body_acc_z',
            'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
            'total_acc_x', 'total_acc_y', 'total_acc_z'
        ]

        train_signals = []
        loaded_train_signals = []
        for signal in signal_types:
            signal_path = UCI_HAR_PATH / 'train' / 'Inertial Signals' / f'{signal}_train.txt'
            if signal_path.exists():
                data = np.loadtxt(signal_path)
                train_signals.append(data)
                loaded_train_signals.append(signal)

        if train_signals:
            self.raw_signals_train = np.stack(train_signals, axis=2)
            shape = self.raw_signals_train.shape
            print(f"✓ Raw signals loaded: {shape}")

        test_signals = []
        loaded_test_signals = []
        for signal in signal_types:
            signal_path = UCI_HAR_PATH / 'test' / 'Inertial Signals' / f'{signal}_test.txt'
            if signal_path.exists():
                data = np.loadtxt(signal_path)
                test_signals.append(data)
                loaded_test_signals.append(signal)

        if test_signals:
            self.raw_signals_test = np.stack(test_signals, axis=2)
            shape = self.raw_signals_test.shape

    def explore_data(self):
        print("\nPHASE 1.2: DATA EXPLORATION")

        unique_train_activities = np.unique(self.y_train)
        for activity_id in unique_train_activities:
            count = np.sum(self.y_train == activity_id)
            percentage = count / len(self.y_train) * 100
            activity_name = ACTIVITIES.get(activity_id, f"Unknown_{activity_id}")

        unique_test_activities = np.unique(self.y_test)
        for activity_id in unique_test_activities:
            count = np.sum(self.y_test == activity_id)
            percentage = count / len(self.y_test) * 100
            activity_name = ACTIVITIES.get(activity_id, f"Unknown_{activity_id}")

        unique_train_subjects = sorted(np.unique(self.subjects_train))
        unique_test_subjects = sorted(np.unique(self.subjects_test))

        train_set = set(unique_train_subjects)
        test_set = set(unique_test_subjects)
        overlap = train_set.intersection(test_set)

        if len(overlap) == 0:
            print("✓ Subject-independent split verified (no overlap)")
        else:
            print(f"⚠️ WARNING: Overlapping subjects found: {sorted(overlap)}")

        for activity_id in unique_train_activities:
            activity_name = ACTIVITIES.get(activity_id, f"Unknown_{activity_id}")
            activity_data = self.X_train[self.y_train == activity_id]

            feature_subset = activity_data[:, :10]
            mean_val = np.mean(feature_subset)
            std_val = np.std(feature_subset)
            min_val = np.min(feature_subset)
            max_val = np.max(feature_subset)
            median_val = np.median(feature_subset)

    def visualize_samples(self):
        if self.raw_signals_train is None:
            print("⚠️  Raw signals not available for visualization")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        unique_activities = np.unique(self.y_train)

        for idx, activity_id in enumerate(unique_activities[:6]):
            activity_name = ACTIVITIES.get(activity_id, f"Unknown_{activity_id}")

            activity_indices = np.where(self.y_train == activity_id)[0]

            if len(activity_indices) > 0:
                sample_idx = activity_indices[0]
                signal = self.raw_signals_train[sample_idx, :, 0]

                axes[idx].plot(signal)
                axes[idx].set_title(f'{activity_name}\n(Sample {sample_idx}, n={len(activity_indices)} total)')
                axes[idx].set_xlabel('Time steps')
                axes[idx].set_ylabel('Acceleration')
                axes[idx].grid(True, alpha=0.3)

                axes[idx].text(0.02, 0.98, f'Mean: {signal.mean():.3f}\nStd: {signal.std():.3f}',
                               transform=axes[idx].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.suptitle('Sample Raw Accelerometer Signals (X-axis) - Data')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'raw_signals_samples.png', dpi=100, bbox_inches='tight')
        print(f"✓ Saved: {RESULTS_DIR / 'raw_signals_samples.png'}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        unique_train = np.unique(self.y_train)
        train_labels = [ACTIVITIES.get(i, f"Unknown_{i}") for i in unique_train]
        train_counts = [np.sum(self.y_train == i) for i in unique_train]

        ax1.bar(train_labels, train_counts, color='steelblue')
        ax1.set_title(f'Training Set Activity Distribution (n={len(self.y_train)})')
        ax1.set_ylabel('Number of Samples')
        ax1.tick_params(axis='x', rotation=45)

        for i, (label, count) in enumerate(zip(train_labels, train_counts)):
            ax1.text(i, count + 10, str(count), ha='center')

        unique_test = np.unique(self.y_test)
        test_labels = [ACTIVITIES.get(i, f"Unknown_{i}") for i in unique_test]
        test_counts = [np.sum(self.y_test == i) for i in unique_test]

        ax2.bar(test_labels, test_counts, color='coral')
        ax2.set_title(f'Test Set Activity Distribution (n={len(self.y_test)})')
        ax2.set_ylabel('Number of Samples')
        ax2.tick_params(axis='x', rotation=45)

        for i, (label, count) in enumerate(zip(test_labels, test_counts)):
            ax2.text(i, count + 5, str(count), ha='center')

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'activity_distribution.png', dpi=100, bbox_inches='tight')
        print(f"✓ Saved: {RESULTS_DIR / 'activity_distribution.png'}")

    def preprocess_data(self):
        print("\nPHASE 1.3: DATA PREPROCESSING")

        print("Normalizing features...")

        feature_min = np.min(self.X_train, axis=0)
        feature_max = np.max(self.X_train, axis=0)

        feature_range = feature_max - feature_min
        zero_range_features = np.where(feature_range == 0)[0]
        if len(zero_range_features) > 0:
            print(f"  ⚠️ Found {len(zero_range_features)} features with zero range")
            feature_range[zero_range_features] = 1

        X_train_norm = 2 * (self.X_train - feature_min) / feature_range - 1
        X_test_norm = 2 * (self.X_test - feature_min) / feature_range - 1

        if X_test_norm.min() < -1.5 or X_test_norm.max() > 1.5:
            print("  ⚠️ Warning: Test data exceeds expected normalized range")

        np.save(DATA_DIR / 'processed' / 'X_train_normalized.npy', X_train_norm)
        np.save(DATA_DIR / 'processed' / 'X_test_normalized.npy', X_test_norm)
        np.save(DATA_DIR / 'processed' / 'y_train.npy', self.y_train)
        np.save(DATA_DIR / 'processed' / 'y_test.npy', self.y_test)

        print("✓ Saved normalized features to data/processed/")

        if self.raw_signals_train is not None:
            np.save(DATA_DIR / 'raw' / 'raw_signals_train.npy', self.raw_signals_train)
            np.save(DATA_DIR / 'raw' / 'raw_signals_test.npy', self.raw_signals_test)
            print("✓ Saved raw signals to data/raw/")

        n_train = len(self.y_train)
        target_val_size = int(0.2 * n_train)

        np.random.seed(42)
        val_indices = []

        unique_activities = np.unique(self.y_train)

        for activity_id in unique_activities:
            activity_indices = np.where(self.y_train == activity_id)[0]
            n_activity = len(activity_indices)
            n_activity_val = int(0.2 * n_activity)

            val_idx = np.random.choice(activity_indices, n_activity_val, replace=False)
            val_indices.extend(val_idx)

            activity_name = ACTIVITIES.get(activity_id, f"Unknown_{activity_id}")

        val_indices = np.array(val_indices)
        train_indices = np.setdiff1d(np.arange(n_train), val_indices)

        train_size = len(train_indices)
        val_size = len(val_indices)

        print(f"✓ Created validation split: {train_size} train, {val_size} validation")

        np.save(DATA_DIR / 'processed' / 'train_indices.npy', train_indices)
        np.save(DATA_DIR / 'processed' / 'val_indices.npy', val_indices)

    def generate_summary(self):

        n_train_samples = len(self.y_train) if self.y_train is not None else 0
        n_test_samples = len(self.y_test) if self.y_test is not None else 0
        n_features = self.X_train.shape[1] if self.X_train is not None else 0

        unique_activities = len(
            np.unique(np.concatenate([self.y_train, self.y_test]))) if self.y_train is not None else 0

        train_subjects = np.unique(self.subjects_train) if self.subjects_train is not None else []
        test_subjects = np.unique(self.subjects_test) if self.subjects_test is not None else []
        total_subjects = len(np.unique(np.concatenate([train_subjects, test_subjects]))) if len(
            train_subjects) > 0 else 0

        processed_files = list(DATA_DIR.glob('processed/*.npy'))
        raw_files = list(DATA_DIR.glob('raw/*.npy'))
        viz_files = list(RESULTS_DIR.glob('*.png'))

        summary = f"""
            PHASE 1 COMPLETION SUMMARY
            {'=' * 60}
            
            DATASET: {n_train_samples} train, {n_test_samples} test, {n_features} features
            RAW SIGNALS: {'Loaded' if self.raw_signals_train is not None else 'Not loaded'}
            FILES CREATED: {len(processed_files)} processed, {len(raw_files)} raw, {len(viz_files)} visualizations
            
            {'=' * 60}
            """
        print(summary)

        with open(RESULTS_DIR / 'phase1_summary.txt', 'w') as f:
            f.write(summary)


def main():
    (DATA_DIR / 'raw').mkdir(parents=True, exist_ok=True)
    (DATA_DIR / 'processed').mkdir(parents=True, exist_ok=True)

    phase1 = Phase1DataPreparation()

    phase1.load_dataset()
    phase1.load_raw_signals()

    phase1.explore_data()
    phase1.visualize_samples()

    phase1.preprocess_data()

    phase1.generate_summary()

    return phase1


if __name__ == "__main__":
    phase1_results = main()
