import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    roc_curve, 
    roc_auc_score,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score
)

warnings.filterwarnings("ignore")

class RFModelEvaluator:
    def __init__(self, predictions_df, labels_df):
        self.predictions_df = predictions_df.copy()
        self.labels_df = labels_df.copy()
        self.evaluation_results = {}
        
    def prepare_data_for_relationship(self, relationship_type):
        prediction_col = f'relationship_{relationship_type}'
        probability_col = f'predicted_probability_{relationship_type}'
        
        if prediction_col not in self.predictions_df.columns:
            raise ValueError(f"Column {prediction_col} not found in predictions dataset")
        if probability_col not in self.predictions_df.columns:
            raise ValueError(f"Column {probability_col} not found in predictions dataset")
        
        pred_data = self.predictions_df[['from_id', 'to_id', prediction_col, probability_col]].copy()
        pred_data = pred_data.rename(columns={
            prediction_col: 'y_pred',
            probability_col: 'y_pred_proba'
        })
        
        labels_data = self.labels_df.copy()
        labels_data['y_true'] = (labels_data['relationship'] == relationship_type).astype(int)
        
        merged_df = pd.merge(
            pred_data, 
            labels_data[['from_id', 'to_id', 'y_true']], 
            on=['from_id', 'to_id'], 
            how='inner'
        )
        
        print(f"DATA PREPARATION FOR {relationship_type.upper()} RELATIONSHIP")
        print(f"Merged dataset shape: {merged_df.shape}")
        print(f"Cases with {relationship_type} relationship: {merged_df['y_true'].sum():,}")
        print(f"Cases without {relationship_type} relationship: {(merged_df['y_true'] == 0).sum():,}")
        
        return merged_df
    
    def evaluate_single_relationship(self, relationship_type, threshold=0.5):
        print(f"EVALUATING {relationship_type.upper()} RELATIONSHIP MODEL")
        
        merged_df = self.prepare_data_for_relationship(relationship_type)
        
        if len(merged_df) == 0:
            print(f"No matching records found for {relationship_type} evaluation")
            return None
        
        y_true = merged_df['y_true'].values
        y_pred = merged_df['y_pred'].values
        y_pred_proba = merged_df['y_pred_proba'].values
        
        if threshold != 0.5:
            y_pred = (y_pred_proba >= threshold).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        try:
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba, pos_label=1)
            auc_score = roc_auc_score(y_true, y_pred_proba)
        except:
            fpr, tpr, roc_thresholds = None, None, None
            auc_score = None
        
    
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            'relationship_type': relationship_type,
            'threshold': threshold,
            'data': merged_df,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'auc_score': auc_score,
            'fpr': fpr,
            'tpr': tpr,
            'roc_thresholds': roc_thresholds
        }
        
        print("CLASSIFICATION REPORT")
        print(classification_report(y_true, y_pred))
        
        print("SUMMARY METRICS")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        if auc_score is not None:
            print(f"ROC AUC: {auc_score:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        
        self._plot_confusion_matrix(y_true, y_pred, relationship_type)
        self._plot_roc_curve(results)
        self._plot_precision_recall_curve(results)
        self._plot_probability_distribution(y_true, y_pred_proba, relationship_type, threshold)
        
        return results
    
    def _plot_confusion_matrix(self, y_true, y_pred, relationship_type):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Relationship', 'Relationship'])
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f'Confusion Matrix - {relationship_type.title()} Relationship', fontsize=14, fontweight='bold')
    
        
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        print(f"\nCONFUSION MATRIX BREAKDOWN")
        print(f"True Negatives (TN): {tn:,}")
        print(f"False Positives (FP): {fp:,}")
        print(f"False Negatives (FN): {fn:,}")
        print(f"True Positives (TP): {tp:,}")
        print(f"Total Cases: {tn + fp + fn + tp:,}")
    
    def _plot_roc_curve(self, results):
        if results['fpr'] is None or results['tpr'] is None:
            print("Cannot plot ROC curve - insufficient data")
            return
        
        plt.figure(figsize=(8, 6))
        plt.plot(results['fpr'], results['tpr'], color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {results["auc_score"]:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {results["relationship_type"].title()} Relationship', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _plot_precision_recall_curve(self, results):
        try:
            precision, recall, _ = precision_recall_curve(results['y_true'], results['y_pred_proba'])
            average_precision = average_precision_score(results['y_true'], results['y_pred_proba'])
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2,
                    label=f'Precision-Recall curve (AP = {average_precision:.4f})')
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title(f'Precision-Recall Curve - {results["relationship_type"].title()} Relationship', 
                     fontsize=14, fontweight='bold')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Cannot plot Precision-Recall curve: {e}")
    
    def _plot_probability_distribution(self, y_true, y_pred_proba, relationship_type, threshold=0.5):
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        tp_mask = (y_true == 1) & (y_pred == 1)
        fp_mask = (y_true == 0) & (y_pred == 1)
        tn_mask = (y_true == 0) & (y_pred == 0)
        fn_mask = (y_true == 1) & (y_pred == 0)
        
        tp_probs = y_pred_proba[tp_mask]
        fp_probs = y_pred_proba[fp_mask]
        tn_probs = y_pred_proba[tn_mask] 
        fn_probs = y_pred_proba[fn_mask]
        
        bins = np.arange(0, 1.1, 0.1)
        bin_labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)]
        
        tp_counts, _ = np.histogram(tp_probs, bins=bins)
        fp_counts, _ = np.histogram(fp_probs, bins=bins)
        tn_counts, _ = np.histogram(tn_probs, bins=bins)
        fn_counts, _ = np.histogram(fn_probs, bins=bins)
        
        total_tp = len(tp_probs)
        total_fp = len(fp_probs) 
        total_tn = len(tn_probs)
        total_fn = len(fn_probs)
        
        tp_pct = (tp_counts / total_tp * 100) if total_tp > 0 else np.zeros_like(tp_counts)
        fp_pct = (fp_counts / total_fp * 100) if total_fp > 0 else np.zeros_like(fp_counts)
        tn_pct = (tn_counts / total_tn * 100) if total_tn > 0 else np.zeros_like(tn_counts)
        fn_pct = (fn_counts / total_fn * 100) if total_fn > 0 else np.zeros_like(fn_counts)
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle(f'Random Forest Model: Prediction Distribution Analysis - {relationship_type.title()}', 
                     fontsize=18, fontweight='bold', y=0.96)
        axes = axes.flatten()
        
        colors = {
            'TP': '#1f77b4',
            'FP': '#ff7f0e',
            'TN': '#2ca02c',
            'FN': '#d62728'
        }
        
        bin_centers = np.arange(len(bin_labels))
        width = 0.7
    
        ax1 = axes[0]
        bars1 = ax1.bar(bin_centers, tp_counts, width, color=colors['TP'], alpha=0.8)
        ax1.set_title(f'True Positives Distribution (Total: {total_tp:,})', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('Predicted Probability Range', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_xticks(bin_centers)
        ax1.set_xticklabels(bin_labels, rotation=45, fontsize=10, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        max_height = max(tp_counts) if max(tp_counts) > 0 else 1
        ax1.set_ylim(0, max_height * 1.3)
        
        for i, (bar, pct) in enumerate(zip(bars1, tp_pct)):
            if pct > 0:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max_height * 0.02,
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        threshold_bin = int(threshold * 10)
        ax1.axvline(threshold_bin - 0.5, color='black', linestyle='--', linewidth=2, 
                    label=f'Threshold ({threshold})')
        ax1.legend(fontsize=10)
        
        ax2 = axes[1]
        bars2 = ax2.bar(bin_centers, fp_counts, width, color=colors['FP'], alpha=0.8)
        ax2.set_title(f'False Positives Distribution (Total: {total_fp:,})', fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('Predicted Probability Range', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_xticks(bin_centers)
        ax2.set_xticklabels(bin_labels, rotation=45, fontsize=10, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        max_height = max(fp_counts) if max(fp_counts) > 0 else 1
        ax2.set_ylim(0, max_height * 1.3)
        
        for i, (bar, pct) in enumerate(zip(bars2, fp_pct)):
            if pct > 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max_height * 0.02,
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax2.axvline(threshold_bin - 0.5, color='black', linestyle='--', linewidth=2)
        
        ax3 = axes[2]
        bars3 = ax3.bar(bin_centers, tn_counts, width, color=colors['TN'], alpha=0.8)
        ax3.set_title(f'True Negatives Distribution (Total: {total_tn:,})', fontsize=14, fontweight='bold', pad=20)
        ax3.set_xlabel('Predicted Probability Range', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.set_xticks(bin_centers)
        ax3.set_xticklabels(bin_labels, rotation=45, fontsize=10, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        max_height = max(tn_counts) if max(tn_counts) > 0 else 1
        ax3.set_ylim(0, max_height * 1.3)
        
        for i, (bar, pct) in enumerate(zip(bars3, tn_pct)):
            if pct > 0:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max_height * 0.02,
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax3.axvline(threshold_bin - 0.5, color='black', linestyle='--', linewidth=2)
        
        ax4 = axes[3]
        bars4 = ax4.bar(bin_centers, fn_counts, width, color=colors['FN'], alpha=0.8)
        ax4.set_title(f'False Negatives Distribution (Total: {total_fn:,})', fontsize=14, fontweight='bold', pad=20)
        ax4.set_xlabel('Predicted Probability Range', fontsize=12)
        ax4.set_ylabel('Count', fontsize=12)
        ax4.set_xticks(bin_centers)
        ax4.set_xticklabels(bin_labels, rotation=45, fontsize=10, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        max_height = max(fn_counts) if max(fn_counts) > 0 else 1
        ax4.set_ylim(0, max_height * 1.3)
        
        for i, (bar, pct) in enumerate(zip(bars4, fn_pct)):
            if pct > 0:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max_height * 0.02,
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax4.axvline(threshold_bin - 0.5, color='black', linestyle='--', linewidth=2)
        
        plt.tight_layout(pad=4.0)
        plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
        plt.show()
        
        print()
        print("PREDICTION DISTRIBUTION BREAKDOWN")
        total_predictions = len(y_true)
        print(f"Total Predictions: {total_predictions:,}")
        print(f"Breakdown by Category:")
        print(f"• True Positives:  {total_tp:,} ({total_tp/total_predictions*100:.1f}%)")
        print(f"• False Positives: {total_fp:,} ({total_fp/total_predictions*100:.1f}%)")
        print(f"• True Negatives:  {total_tn:,} ({total_tn/total_predictions*100:.1f}%)")
        print(f"• False Negatives: {total_fn:,} ({total_fn/total_predictions*100:.1f}%)")
        
        results_df = pd.DataFrame({
            'Probability_Range': bin_labels,
            'TP_Count': tp_counts.astype(int),
            'TP_%': tp_pct.round(1),
            'FP_Count': fp_counts.astype(int), 
            'FP_%': fp_pct.round(1),
            'TN_Count': tn_counts.astype(int),
            'TN_%': tn_pct.round(1),
            'FN_Count': fn_counts.astype(int),
            'FN_%': fn_pct.round(1)
        })
        
        print()
        print("DISTRIBUTION BY PROBABILITY RANGE")
        print(results_df.to_string(index=False))

    def analyze_false_positives(self, relationship_type, top_n=10):
        if relationship_type not in self.evaluation_results:
            print(f"No evaluation results found for {relationship_type}")
            return
        
        results = self.evaluation_results[relationship_type]
        data = results['data']
        
        fp_mask = (data['y_pred'] == 1) & (data['y_true'] == 0)
        false_positives = data[fp_mask].copy()
        
        if len(false_positives) == 0:
            print(f"No false positives found for {relationship_type}")
            return
        
        false_positives = false_positives.sort_values('y_pred_proba', ascending=False)
        
        print(f"FALSE POSITIVE ANALYSIS - {relationship_type.upper()} RELATIONSHIP")
        print(f"Total False Positives: {len(false_positives)}")
        print(f"Average Prediction Probability: {false_positives['y_pred_proba'].mean():.4f}")
        print(f"Probability Range: {false_positives['y_pred_proba'].min():.4f} - {false_positives['y_pred_proba'].max():.4f}")
        
        print(f"\nTop {min(top_n, len(false_positives))} High-Confidence False Positives:")
        display_cols = ['from_id', 'to_id', 'y_pred_proba']
        print(false_positives[display_cols].head(top_n).to_string(index=False))
        
        return false_positives
    
    def analyze_false_negatives(self, relationship_type, top_n=10):
        if relationship_type not in self.evaluation_results:
            print(f"No evaluation results found for {relationship_type}")
            return
        
        results = self.evaluation_results[relationship_type]
        data = results['data']
        
        fn_mask = (data['y_pred'] == 0) & (data['y_true'] == 1)
        false_negatives = data[fn_mask].copy()
        
        if len(false_negatives) == 0:
            print(f"No false negatives found for {relationship_type}")
            return
        
        false_negatives = false_negatives.sort_values('y_pred_proba', ascending=True)
        
        print(f"FALSE NEGATIVE ANALYSIS - {relationship_type.upper()} RELATIONSHIP")
        print(f"Total False Negatives: {len(false_negatives)}")
        print(f"Average Prediction Probability: {false_negatives['y_pred_proba'].mean():.4f}")
        print(f"Probability Range: {false_negatives['y_pred_proba'].min():.4f} - {false_negatives['y_pred_proba'].max():.4f}")
        
        print(f"\nTop {min(top_n, len(false_negatives))} Low-Confidence False Negatives:")
        display_cols = ['from_id', 'to_id', 'y_pred_proba']
        print(false_negatives[display_cols].head(top_n).to_string(index=False))
        
        return false_negatives


def evaluate_rf_models(predictions_df, labels_df, relationship_types=None, threshold=0.5):
    import warnings
    warnings.filterwarnings("ignore")
    
    print("Starting Random Forest Model Evaluation...")
    
    evaluator = RFModelEvaluator(predictions_df, labels_df)
    
    if relationship_types is None:
        relationship_cols = [col for col in predictions_df.columns if col.startswith('relationship_')]
        relationship_types = [col.replace('relationship_', '') for col in relationship_cols]
        print(f"Auto-detected relationship types: {relationship_types}")
    
    results = {}
    for rel_type in relationship_types:
        try:
            result = evaluator.evaluate_single_relationship(rel_type, threshold=threshold)
            if result is not None:
                results[rel_type] = result
            else:
                print(f"Failed to evaluate {rel_type}")
        except Exception as e:
            print(f"Error evaluating {rel_type}: {str(e)}")
            continue
    
    evaluator.evaluation_results = results
    
    print("Model evaluation completed.")
    
    return evaluator

# Usage example
predictions_df = pd.read_csv('/Users/abhinavpundir/Downloads/family_linkage_models-main-2/data/processed/test_predictions_sibling.csv')
labels_df = pd.read_csv('/Users/abhinavpundir/Downloads/family_linkage_models-main-2/data/test/testing_labels.csv')

# Evaluate specific relationship
evaluator = evaluate_rf_models(predictions_df, labels_df, relationship_types=['sibling'])