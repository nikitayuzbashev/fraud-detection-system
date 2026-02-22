"""Model traning for fraud detection"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import xgboost as xgb
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE    # Handle imbalanced data
from imblearn.pipeline import Pipeline as ImbPipeline   # Pipeline to work with SMOTE

from ..data.preprocessor import create_preprocessing_pipeline
from ..utils.config import Config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class FraudDetectionTrainer:
    """Trainer for fraud detection models"""
    
    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        use_smote: bool = True
    ):
        self.model_config = model_config or Config.get_model_config()
        self.test_size = test_size
        self.random_state = random_state
        self.use_smote = use_smote
        
        self.model = None
        self.pipeline = None
        self.preprocessing_pipeline = None
    
    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        logger.info("Splitting data into train and test sets")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y  
            # Imbalanced dataset
            # mostly legitimate transactions, stratify to ensure train and test have same proportion of fraud cases 
        )
        
        logger.info(f"Train: {len(X_train)} samples ({y_train.sum()} frauds)")
        logger.info(f"Test: {len(X_test)} samples ({y_test.sum()} frauds)")
        
        return X_train, X_test, y_train, y_test
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> xgb.XGBClassifier:
        """Train the fraud detection model"""
        logger.info("Starting model training")
        logger.info(f"Model config: {self.model_config}")
        
        from ..data.preprocessor import FeatureEngineer
        from sklearn.preprocessing import StandardScaler
        
        steps = [
            ('feature_engineering', FeatureEngineer()),
            ('scaler', StandardScaler())
        ]
        
        if self.use_smote:
            smote = SMOTE(random_state=self.random_state)
            steps.append(('smote', smote))
            logger.info("SMOTE enabled for handling imbalanced data")
        
        self.model = xgb.XGBClassifier(**self.model_config)
        steps.append(('classifier', self.model))
        
        self.pipeline = ImbPipeline(steps)
        
        logger.info("Training XGBoost model...")
        self.pipeline.fit(X_train, y_train)
        logger.info("Model training complete")
    
        return self.model
    
    def save_model(self, model_path: Optional[Path] = None) -> Path:
        """Save trained model to disk."""
        if self.pipeline is None:
            raise ValueError("No model to save.")
        
        save_path = model_path or Config.MODEL_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {save_path}")
        joblib.dump(self.pipeline, save_path)
        logger.info("Model saved successfully")
        
        return save_path
    
    @staticmethod
    def load_model(model_path: Optional[Path] = None):
        """Load trained model from disk"""
        load_path = model_path or Config.MODEL_PATH
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found at {load_path}")
        
        logger.info(f"Loading model from {load_path}")
        return joblib.load(load_path)