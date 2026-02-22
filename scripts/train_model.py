"""Script to train the fraud detection model"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_data, split_features_target
from src.models.trainer import FraudDetectionTrainer
from src.models.evaluator import ModelEvaluator
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("FRAUD DETECTION MODEL TRAINING")
    logger.info("=" * 60)
    
    # Load data
    logger.info("\n1. Loading Data:")
    try:
        df = load_data()
    except FileNotFoundError:
        logger.error("Data file not found. Creating synthetic data:")
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        n_samples = 10000
        
        data = {'Time': np.random.uniform(0, 172800, n_samples)}
        for i in range(1, 29):
            data[f'V{i}'] = np.random.randn(n_samples)
        data['Amount'] = np.random.exponential(88, n_samples)
        data['Class'] = np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
        
        df = pd.DataFrame(data)
        logger.info(f"Created synthetic dataset: {df.shape}")
    
    # Split features and target
    X, y = split_features_target(df)
    
    # Initialize trainer
    logger.info("\n2. Initializing Trainer:")
    trainer = FraudDetectionTrainer()
    
    # Prepare data
    logger.info("\n3. Preparing Data:")
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
    
    # Train model
    logger.info("\n4. Training Model:")
    trainer.train(X_train, y_train)
    
    # Evaluate
    logger.info("\n5. Evaluating Model:")
    evaluator = ModelEvaluator()
    
    y_pred = trainer.pipeline.predict(X_test)
    y_proba = trainer.pipeline.predict_proba(X_test)[:, 1]
    
    metrics = evaluator.evaluate(y_test, y_pred, y_proba)
    evaluator.print_report(y_test, y_pred)
    
    # Save model
    logger.info("\n6. Saving Model:")
    model_path = trainer.save_model()
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"\nKey Metrics:")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()