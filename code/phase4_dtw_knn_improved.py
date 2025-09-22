import numpy as np
import pandas as pd
import time
import json
import pickle
import os
import sys
import warnings
from pathlib import Path
from collections import Counter
from scipy.stats import mode
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import contextlib
import io

warnings.filterwarnings('ignore')


@contextlib.contextmanager
def suppress_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


try:
    with suppress_output():
        from dtaidistance import dtw
    DTAIDISTANCE_AVAILABLE = True
except ImportError:
    DTAIDISTANCE_AVAILABLE = False
    print("⚠️ dtaidistance not available, using custom implementation")


class ImprovedDTWkNN:

    def __init__(self, k=7, window_ratio=0.1, use_pca=False, n_components=50,
                 voting='weighted', normalize=True, multi_channel='independent',
                 early_abandon=True, lb_keogh=True, adaptive_k=True):
        self.k = k
        self.window_ratio = window_ratio
        self.use_pca = use_pca
        self.n_components = n_components
        self.voting = voting
        self.normalize = normalize
        self.multi_channel = multi_channel
        self.early_abandon = early_abandon
        self.lb_keogh = lb_keogh
        self.adaptive_k = adaptive_k

        self.X_train = None
        self.y_train = None
        self.distance_matrix = None
        self.scaler = StandardScaler()
        self.pca = None

        self.activity_names = {
            1: "WALKING",
            2: "WALKING_UPSTAIRS",
            3: "WALKING_DOWNSTAIRS",
            4: "SITTING",
            5: "STANDING",
            6: "LAYING"
        }

    def z_normalize(self, x):
        if len(x.shape) == 1:
            mean = np.mean(x)
            std = np.std(x)
            if std < 1e-6:
                return x - mean
            return (x - mean) / std
        else:
            normalized = np.zeros_like(x)
            for i in range(x.shape[1]):
                mean = np.mean(x[:, i])
                std = np.std(x[:, i])
                if std < 1e-6:
                    normalized[:, i] = x[:, i] - mean
                else:
                    normalized[:, i] = (x[:, i] - mean) / std
            return normalized

    def lb_keogh_lower_bound(self, s1, s2, window):
        n = len(s1)
        lb_sum = 0

        for i in range(n):
            lower = max(0, i - window)
            upper = min(n - 1, i + window)

            min_val = np.min(s2[lower:upper + 1])
            max_val = np.max(s2[lower:upper + 1])

            if s1[i] > max_val:
                lb_sum += (s1[i] - max_val) ** 2
            elif s1[i] < min_val:
                lb_sum += (min_val - s1[i]) ** 2

        return np.sqrt(lb_sum)

    def dtw_distance_multi_channel(self, x, y, channel_weights=None):
        if len(x.shape) == 1:
            return self.compute_dtw_distance_1d(x, y)

        n_channels = x.shape[1]

        if channel_weights is None:
            channel_weights = np.array([1.0, 1.0, 1.0,
                                        0.8, 0.8, 0.8,
                                        1.2, 1.2, 1.2])
            channel_weights = channel_weights[:n_channels]
            channel_weights /= channel_weights.sum()

        if self.multi_channel == 'independent':
            distances = []
            for i in range(n_channels):
                if self.normalize:
                    x_ch = self.z_normalize(x[:, i])
                    y_ch = self.z_normalize(y[:, i])
                else:
                    x_ch = x[:, i]
                    y_ch = y[:, i]

                dist = self.compute_dtw_distance_1d(x_ch, y_ch)
                distances.append(dist * channel_weights[i])

            return np.sum(distances)

        elif self.multi_channel == 'dependent':
            if self.normalize:
                x = self.z_normalize(x)
                y = self.z_normalize(y)

            if DTAIDISTANCE_AVAILABLE:
                with suppress_output():
                    from dtaidistance import dtw_ndim
                    distance = dtw_ndim.distance(x, y, window=self.window)
            else:
                distance = self.compute_dtw_distance_ndim(x, y)

            return distance

        else:
            x_mag = np.linalg.norm(x, axis=1)
            y_mag = np.linalg.norm(y, axis=1)

            if self.normalize:
                x_mag = self.z_normalize(x_mag)
                y_mag = self.z_normalize(y_mag)

            return self.compute_dtw_distance_1d(x_mag, y_mag)

    def compute_dtw_distance_1d(self, x, y):
        n = len(x)
        window = int(n * self.window_ratio)

        if self.lb_keogh:
            lb = self.lb_keogh_lower_bound(x, y, window)

        if DTAIDISTANCE_AVAILABLE:
            with suppress_output():
                distance = dtw.distance(x, y, window=window, use_c=True)
        else:
            distance = self.dtw_distance_custom_1d(x, y, window)

        return distance

    def dtw_distance_custom_1d(self, x, y, window):
        n, m = len(x), len(y)

        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0

        best_so_far = np.inf

        for i in range(1, n + 1):
            min_cost_in_row = np.inf

            for j in range(max(1, i - window), min(m + 1, i + window + 1)):
                cost[i, j] = abs(x[i - 1] - y[j - 1]) + min(
                    cost[i - 1, j],
                    cost[i, j - 1],
                    cost[i - 1, j - 1]
                )

                min_cost_in_row = min(min_cost_in_row, cost[i, j])

            if self.early_abandon and min_cost_in_row > best_so_far:
                return np.inf

        return cost[n, m]

    def compute_dtw_distance_ndim(self, x, y):
        n, m = len(x), len(y)
        window = int(n * self.window_ratio)

        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(max(1, i - window), min(m + 1, i + window + 1)):
                dist = np.linalg.norm(x[i - 1] - y[j - 1])
                cost[i, j] = dist + min(
                    cost[i - 1, j],
                    cost[i, j - 1],
                    cost[i - 1, j - 1]
                )

        return cost[n, m]

    def fit(self, X_train, y_train, precompute_distances=True):
        print("Fitting DTW-kNN...")

        self.X_train = X_train
        self.y_train = y_train
        self.n_train = len(X_train)
        self.window = int(X_train.shape[1] * self.window_ratio)

        if self.use_pca and X_train.shape[2] > 1:
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=self.n_components)

            X_reshaped = X_train.reshape(X_train.shape[0], -1)
            X_transformed = self.pca.fit_transform(X_reshaped)

            new_shape = (X_train.shape[0], X_train.shape[1], self.n_components)
            self.X_train = X_transformed.reshape(new_shape)

            print(f"  PCA variance: {self.pca.explained_variance_ratio_.sum():.4f}")

        if precompute_distances and self.n_train <= 10000:
            print("Precomputing distances...")
            self.distance_matrix = self.compute_distance_matrix()
            print("✓ Distance matrix computed")

        print(f"✓ Fitted with {self.n_train} samples")

    def compute_distance_matrix(self):
        n = self.n_train
        distances = np.zeros((n, n))

        total_comparisons = n * (n - 1) // 2
        computed = 0
        start_time = time.time()

        for i in range(n):
            for j in range(i + 1, n):
                dist = self.dtw_distance_multi_channel(
                    self.X_train[i], self.X_train[j]
                )
                distances[i, j] = dist
                distances[j, i] = dist

                computed += 1

                if computed % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = computed / elapsed
                    remaining = (total_comparisons - computed) / rate
                    print(f"  {100 * computed / total_comparisons:.1f}% - ETA: {remaining / 60:.1f}m")

        return distances

    def predict_sample(self, test_sample, train_data=None, train_labels=None):
        if train_data is None:
            train_data = self.X_train
            train_labels = self.y_train

        n_train = len(train_data)

        if self.pca is not None:
            test_reshaped = test_sample.reshape(1, -1)
            test_transformed = self.pca.transform(test_reshaped)
            test_sample = test_transformed.reshape(test_sample.shape[0], -1)

        distances = np.zeros(n_train)

        for j in range(n_train):
            distances[j] = self.dtw_distance_multi_channel(
                test_sample, train_data[j]
            )

        if self.adaptive_k:
            sorted_distances = np.sort(distances)
            gaps = np.diff(sorted_distances[:self.k * 2])

            if len(gaps) > 0:
                elbow = np.argmax(gaps) + 1
                k_adaptive = min(max(3, elbow), self.k)
            else:
                k_adaptive = self.k
        else:
            k_adaptive = self.k

        nearest_indices = np.argsort(distances)[:k_adaptive]
        nearest_distances = distances[nearest_indices]
        nearest_labels = train_labels[nearest_indices]

        if self.voting == 'weighted':
            epsilon = 1e-10
            weights = 1.0 / (nearest_distances + epsilon)
            weights /= weights.sum()

            weighted_votes = {}
            for label, weight in zip(nearest_labels, weights):
                if label not in weighted_votes:
                    weighted_votes[label] = 0
                weighted_votes[label] += weight

            prediction = max(weighted_votes, key=weighted_votes.get)
            confidence = weighted_votes[prediction]

        else:
            label_counts = Counter(nearest_labels)
            prediction = label_counts.most_common(1)[0][0]
            confidence = label_counts[prediction] / k_adaptive

        neighbor_info = {
            'indices': nearest_indices,
            'distances': nearest_distances,
            'labels': nearest_labels,
            'k_used': k_adaptive
        }

        return prediction, confidence, neighbor_info

    def predict(self, X_test):
        n_test = len(X_test)
        predictions = np.zeros(n_test, dtype=int)
        confidences = np.zeros(n_test)

        print(f"Predicting {n_test} samples...")
        start_time = time.time()

        for i in range(n_test):
            pred, conf, _ = self.predict_sample(X_test[i])
            predictions[i] = pred
            confidences[i] = conf

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 1
                remaining = (n_test - i - 1) / rate
                print(f"  {i + 1}/{n_test} - ETA: {remaining / 60:.1f}m")
                sys.stdout.flush()

        total_time = time.time() - start_time
        print(f"✓ Predictions completed in {total_time:.2f}s")

        return predictions, confidences


def load_data():
    print("Loading UCI-HAR dataset...")

    data_dir = Path("../UCI HAR Dataset")

    def load_inertial_signals(subset='train'):
        signals_dir = data_dir / subset / 'Inertial Signals'

        channels = []
        signal_names = [
            'body_acc_x', 'body_acc_y', 'body_acc_z',
            'total_acc_x', 'total_acc_y', 'total_acc_z',
            'body_gyro_x', 'body_gyro_y', 'body_gyro_z'
        ]

        for signal_name in signal_names:
            signal_file = signals_dir / f'{signal_name}_{subset}.txt'
            signal = pd.read_csv(signal_file, delim_whitespace=True, header=None).values
            channels.append(signal)

        X = np.stack(channels, axis=2)

        y = pd.read_csv(data_dir / subset / f'y_{subset}.txt', header=None).values.flatten()

        return X, y

    X_train, y_train = load_inertial_signals('train')
    print(f"  Training: {X_train.shape}")

    X_test, y_test = load_inertial_signals('test')
    print(f"  Test: {X_test.shape}")

    return X_train, y_train, X_test, y_test


def main():
    print("IMPROVED DTW-KNN CLASSIFIER")

    X_train, y_train, X_test, y_test = load_data()

    subset_size = 7352
    test_subset = 2947

    print(f"Using: {subset_size} train, {test_subset} test samples")

    X_train_subset = X_train[:subset_size]
    y_train_subset = y_train[:subset_size]
    X_test_subset = X_test[:test_subset]
    y_test_subset = y_test[:test_subset]

    classifier = ImprovedDTWkNN(
        k=7,
        window_ratio=0.15,
        voting='weighted',
        normalize=True,
        multi_channel='independent',
        early_abandon=True,
        lb_keogh=True,
        adaptive_k=True
    )

    start_time = time.time()
    classifier.fit(X_train_subset, y_train_subset, precompute_distances=False)
    train_time = time.time() - start_time

    predictions, confidences = classifier.predict(X_test_subset)

    accuracy = accuracy_score(y_test_subset, predictions)
    precision = precision_score(y_test_subset, predictions, average='weighted')
    recall = recall_score(y_test_subset, predictions, average='weighted')
    f1 = f1_score(y_test_subset, predictions, average='weighted')

    print(f"\n✓ Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    os.makedirs('results/phase4_improved', exist_ok=True)

    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'avg_confidence': float(np.mean(confidences)),
        'train_samples': subset_size,
        'test_samples': test_subset,
        'configuration': {
            'k': classifier.k,
            'window_ratio': classifier.window_ratio,
            'voting': classifier.voting,
            'normalize': classifier.normalize,
            'multi_channel': classifier.multi_channel,
            'adaptive_k': classifier.adaptive_k
        }
    }

    with open('results/phase4_improved/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    summary = f"""
IMPROVED DTW-KNN RESULTS
Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)
F1-Score: {f1:.4f}
Samples: {subset_size} train, {test_subset} test
"""

    print(summary)

    with open('results/phase4_improved/summary.txt', 'w') as f:
        f.write(summary)

    print("✓ Results saved to results/phase4_improved/")

    return results


if __name__ == "__main__":
    import sys

    results = main()
