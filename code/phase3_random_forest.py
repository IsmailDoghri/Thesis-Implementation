import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import os
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results' / 'phase3'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ACTIVITIES = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING'
}


class RandomForestHAR:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            oob_score=True,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_importances = None
        self.predictions = None
        self.train_time = None
        self.inference_time = None

    def load_data(self):
        print("PHASE 3: RANDOM FOREST")
        print("Loading data...")

        train_path = DATA_DIR / 'processed' / 'X_train_normalized.npy'
        test_path = DATA_DIR / 'processed' / 'X_test_normalized_clipped.npy'

        if not train_path.exists():
            train_path = BASE_DIR / 'results' / 'phase2' / 'X_train_normalized.npy'
            test_path = BASE_DIR / 'results' / 'phase2' / 'X_test_normalized_clipped.npy'

        self.X_train = np.load(train_path)
        self.X_test = np.load(test_path)

        self.y_train = np.load(DATA_DIR / 'processed' / 'y_train.npy')
        self.y_test = np.load(DATA_DIR / 'processed' / 'y_test.npy')

        self.train_indices = np.load(DATA_DIR / 'processed' / 'train_indices.npy')
        self.val_indices = np.load(DATA_DIR / 'processed' / 'val_indices.npy')

        print(f"✓ Loaded: {self.X_train.shape[0]} train, {self.X_test.shape[0]} test samples")

        for activity_id in np.unique(self.y_train):
            train_count = np.sum(self.y_train == activity_id)
            test_count = np.sum(self.y_test == activity_id)
            activity_name = ACTIVITIES[activity_id]

    def train(self):
        print("Training model...")
        start_time = time.time()

        self.model.fit(self.X_train, self.y_train)

        self.train_time = time.time() - start_time
        print(f"✓ Training completed in {self.train_time:.2f}s")

        if self.model.oob_score:
            print(f"  OOB Score: {self.model.oob_score_:.4f}")

        self.feature_importances = self.model.feature_importances_

    def evaluate(self):
        print("Evaluating...")
        start_time = time.time()
        self.predictions = self.model.predict(self.X_test)
        self.inference_time = time.time() - start_time

        accuracy = accuracy_score(self.y_test, self.predictions)
        precision = precision_score(self.y_test, self.predictions, average='weighted')
        recall = recall_score(self.y_test, self.predictions, average='weighted')
        f1 = f1_score(self.y_test, self.predictions, average='weighted')

        print(f"✓ Accuracy: {accuracy:.4f}")

        for activity_id in sorted(np.unique(self.y_test)):
            activity_name = ACTIVITIES[activity_id]
            mask = self.y_test == activity_id
            pred_mask = self.predictions == activity_id

            tp = np.sum((self.y_test == activity_id) & (self.predictions == activity_id))
            fp = np.sum((self.y_test != activity_id) & (self.predictions == activity_id))
            fn = np.sum((self.y_test == activity_id) & (self.predictions != activity_id))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_class = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            support = np.sum(mask)

        self.confusion_matrix = confusion_matrix(self.y_test, self.predictions)

        self.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'train_time': self.train_time,
            'inference_time': self.inference_time,
            'inference_per_sample_ms': self.inference_time / len(self.X_test) * 1000,
            'oob_score': self.model.oob_score_ if self.model.oob_score else None,
            'n_features': self.X_train.shape[1],
            'n_train_samples': len(self.X_train),
            'n_test_samples': len(self.X_test)
        }

        return self.results

    def analyze_feature_importance(self):
        uci_path = BASE_DIR.parent / 'UCI HAR Dataset' / 'features.txt'
        feature_names = []

        if uci_path.exists():
            with open(uci_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        feature_names.append(' '.join(parts[1:]))
        else:
            feature_names = [f'Feature_{i}' for i in range(len(self.feature_importances))]

        importance_indices = np.argsort(self.feature_importances)[::-1]

        n_top = 20

        top_features = []
        for i in range(min(n_top, len(importance_indices))):
            idx = importance_indices[i]
            importance = self.feature_importances[idx]
            name = feature_names[idx] if idx < len(feature_names) else f'Feature_{idx}'
            top_features.append({
                'rank': i + 1,
                'index': int(idx),
                'name': name,
                'importance': float(importance)
            })

        time_domain_keywords = ['mean', 'std', 'mad', 'max', 'min', 'sma', 'energy',
                                'iqr', 'entropy', 'arCoeff', 'correlation', 'angle']
        freq_domain_keywords = ['Freq', 'freq']

        time_importance = 0
        freq_importance = 0

        for idx, importance in enumerate(self.feature_importances):
            if idx < len(feature_names):
                name = feature_names[idx]
                if any(keyword in name for keyword in freq_domain_keywords):
                    freq_importance += importance
                else:
                    time_importance += importance

        signal_types = ['BodyAcc', 'GravityAcc', 'BodyGyro', 'BodyAccJerk', 'BodyGyroJerk']
        signal_importance = {}

        for signal_type in signal_types:
            importance_sum = 0
            for idx, importance in enumerate(self.feature_importances):
                if idx < len(feature_names) and signal_type in feature_names[idx]:
                    importance_sum += importance
            signal_importance[signal_type] = importance_sum

        self.feature_analysis = {
            'top_features': top_features,
            'time_domain_importance': float(time_importance),
            'freq_domain_importance': float(freq_importance),
            'signal_importance': {k: float(v) for k, v in signal_importance.items()}
        }

        return self.feature_analysis

    def visualize_results(self):
        fig = plt.figure(figsize=(16, 12))

        ax1 = plt.subplot(2, 3, 1)
        sns.heatmap(self.confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[ACTIVITIES[i] for i in sorted(ACTIVITIES.keys())],
                    yticklabels=[ACTIVITIES[i] for i in sorted(ACTIVITIES.keys())])
        ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')

        ax2 = plt.subplot(2, 3, 2)
        cm_normalized = self.confusion_matrix.astype('float') / self.confusion_matrix.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=[ACTIVITIES[i] for i in sorted(ACTIVITIES.keys())],
                    yticklabels=[ACTIVITIES[i] for i in sorted(ACTIVITIES.keys())])
        ax2.set_title('Normalized Confusion Matrix', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')

        ax3 = plt.subplot(2, 3, 3)
        per_class_acc = []
        class_names = []
        for i in range(len(self.confusion_matrix)):
            acc = self.confusion_matrix[i, i] / self.confusion_matrix[i].sum()
            per_class_acc.append(acc)
            class_names.append(ACTIVITIES[i + 1])

        bars = ax3.bar(class_names, per_class_acc, color='steelblue')
        ax3.set_title('Per-Class Accuracy', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Accuracy')
        ax3.set_ylim([0, 1.1])
        ax3.tick_params(axis='x', rotation=45)

        for bar, acc in zip(bars, per_class_acc):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{acc:.3f}', ha='center', va='bottom')

        ax4 = plt.subplot(2, 3, 4)
        importance_indices = np.argsort(self.feature_importances)[::-1][:20]
        top_importances = self.feature_importances[importance_indices]

        ax4.barh(range(20), top_importances[::-1], color='forestgreen')
        ax4.set_title('Top 20 Feature Importances', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Importance')
        ax4.set_ylabel('Feature Rank')
        ax4.set_yticks(range(20))
        ax4.set_yticklabels([f'#{20 - i}' for i in range(20)])

        ax5 = plt.subplot(2, 3, 5)
        if hasattr(self, 'feature_analysis'):
            domains = ['Time Domain', 'Frequency Domain']
            importances = [self.feature_analysis['time_domain_importance'],
                           self.feature_analysis['freq_domain_importance']]
            colors = ['#2E86AB', '#A23B72']

            wedges, texts, autotexts = ax5.pie(importances, labels=domains, colors=colors,
                                               autopct='%1.1f%%', startangle=90)
            ax5.set_title('Feature Importance by Domain', fontsize=12, fontweight='bold')

        ax6 = plt.subplot(2, 3, 6)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [self.results['accuracy'], self.results['precision'],
                  self.results['recall'], self.results['f1_score']]

        bars = ax6.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax6.set_title('Performance Metrics', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Score')
        ax6.set_ylim([0.8, 1.0])
        ax6.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width() / 2., height + 0.002,
                     f'{val:.4f}', ha='center', va='bottom')

        plt.suptitle('Phase 3: Random Forest Results', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        plt.savefig(RESULTS_DIR / 'phase3_random_forest_results.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization: phase3_random_forest_results.png")

        self.create_feature_importance_plot()

    def create_feature_importance_plot(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax1 = axes[0, 0]
        sorted_importances = np.sort(self.feature_importances)[::-1]
        cumsum_importance = np.cumsum(sorted_importances)

        ax1.plot(cumsum_importance, 'b-', linewidth=2)
        ax1.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% threshold')
        ax1.set_xlabel('Number of Features')
        ax1.set_ylabel('Cumulative Importance')
        ax1.set_title('Cumulative Feature Importance')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        n_features_90 = np.argmax(cumsum_importance >= 0.9) + 1
        ax1.text(n_features_90, 0.9, f' {n_features_90} features',
                 verticalalignment='bottom')

        ax2 = axes[0, 1]
        ax2.hist(self.feature_importances, bins=50, color='forestgreen', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Importance Value')
        ax2.set_ylabel('Number of Features')
        ax2.set_title('Feature Importance Distribution')
        ax2.axvline(x=np.mean(self.feature_importances), color='red',
                    linestyle='--', label=f'Mean: {np.mean(self.feature_importances):.4f}')
        ax2.legend()

        ax3 = axes[1, 0]
        if hasattr(self, 'feature_analysis'):
            signal_data = self.feature_analysis['signal_importance']
            signals = list(signal_data.keys())
            importances = list(signal_data.values())

            bars = ax3.bar(signals, importances, color='teal')
            ax3.set_title('Importance by Signal Type')
            ax3.set_ylabel('Total Importance')
            ax3.tick_params(axis='x', rotation=45)

            for bar, imp in zip(bars, importances):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{imp * 100:.1f}%', ha='center', va='bottom')

        ax4 = axes[1, 1]
        if hasattr(self, 'feature_analysis'):
            top_10 = self.feature_analysis['top_features'][:10]
            names = [f['name'][:30] for f in top_10]
            importances = [f['importance'] for f in top_10]

            y_pos = np.arange(len(names))
            ax4.barh(y_pos, importances, color='coral')
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(names, fontsize=8)
            ax4.set_xlabel('Importance')
            ax4.set_title('Top 10 Features (Detailed)')
            ax4.invert_yaxis()

        plt.suptitle('Random Forest Feature Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        plt.savefig(RESULTS_DIR / 'phase3_feature_analysis.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved feature analysis: phase3_feature_analysis.png")

    def save_results(self):
        import joblib
        model_path = RESULTS_DIR / 'random_forest_model.joblib'
        joblib.dump(self.model, model_path)
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Saved model ({model_size_mb:.2f} MB)")

        np.save(RESULTS_DIR / 'predictions.npy', self.predictions)

        np.save(RESULTS_DIR / 'feature_importances.npy', self.feature_importances)

        json_results = {
            'model_config': {
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth,
                'min_samples_split': self.model.min_samples_split,
                'min_samples_leaf': self.model.min_samples_leaf,
                'max_features': self.model.max_features,
                'random_state': self.model.random_state
            },
            'performance': self.results,
            'feature_analysis': self.feature_analysis if hasattr(self, 'feature_analysis') else None,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'model_size_mb': model_size_mb,
            'n_features_for_90_percent': int(np.argmax(np.cumsum(np.sort(self.feature_importances)[::-1]) >= 0.9) + 1)
        }

        with open(RESULTS_DIR / 'phase3_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)

        return json_results


def generate_comprehensive_summary(rf_model, results):
    summary = f"""
PHASE 3: RANDOM FOREST SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

Performance: {results['performance']['accuracy']:.4f} accuracy
Training: {results['performance']['train_time']:.2f}s
Inference: {results['performance']['inference_per_sample_ms']:.4f}ms per sample
Model Size: {results['model_size_mb']:.2f} MB

Top Confused Pairs:
"""

    cm = np.array(results['confusion_matrix'])
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    confused_pairs = []

    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((i + 1, j + 1, cm[i, j]))

    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    for i in range(min(3, len(confused_pairs))):
        summary += f"  {ACTIVITIES[confused_pairs[i][0]]} → {ACTIVITIES[confused_pairs[i][1]]}: {confused_pairs[i][2]} errors\n"

    summary += f"""
================================================================================
Phase 3 Completed Successfully
"""

    return summary


def main():
    rf_model = RandomForestHAR()

    rf_model.load_data()

    rf_model.train()

    results = rf_model.evaluate()

    rf_model.analyze_feature_importance()

    rf_model.visualize_results()

    json_results = rf_model.save_results()

    summary = generate_comprehensive_summary(rf_model, json_results)

    summary_path = RESULTS_DIR / 'phase3_comprehensive_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)

    print(f"✓ Saved summary to {summary_path}")

    print("✓ PHASE 3 COMPLETED")

    return rf_model, json_results


if __name__ == '__main__':
    rf_model, results = main()
