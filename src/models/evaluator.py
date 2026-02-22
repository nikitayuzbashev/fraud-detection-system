"""Model evaluation for fraud detection"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelEvaluator:
    """Evaluator for fraud detection models"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.metrics = {}
    
    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None
    ) -> Dict[str, Any]:
        """Evaluate model performance."""
        logger.info("Evaluating model performance")
        
        self.metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_proba is not None:
            self.metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        self.metrics['confusion_matrix'] = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        
        self.metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        logger.info(f"Precision: {self.metrics['precision']:.4f}")
        logger.info(f"Recall: {self.metrics['recall']:.4f}")
        logger.info(f"F1-Score: {self.metrics['f1_score']:.4f}")
        if y_proba is not None:
            logger.info(f"ROC-AUC: {self.metrics['roc_auc']:.4f}")
        
        return self.metrics
    
    def print_report(self, y_true: pd.Series, y_pred: np.ndarray) -> str:
        """Generate classification report."""
        report = classification_report(
            y_true, y_pred,
            target_names=['Legitimate', 'Fraud'],
            digits=4
        )
        logger.info(f"\n{report}")
        return report