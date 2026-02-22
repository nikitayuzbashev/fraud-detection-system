"""Feature engineering and preprocessing for fraud detection."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Optional

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for fraud detection feature engineering.
    
    Creates temporal features, amount transformations, and interactions
    to help identify fraudulent patterns.
    """
    
    def __init__(self, create_time_features: bool = True):
        self.create_time_features = create_time_features
        self.feature_names_ = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit transformer (no-op, required for sklearn compatibility)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from transaction data.
        
        Args:
            X: Input features
        
        Returns:
            DataFrame with original + engineered features
        """
        X = X.copy()
        
        # Time features with cyclical encoding
        if self.create_time_features and 'Time' in X.columns:
            X['Hour'] = (X['Time'] / 3600) % 24
            # Cyclical encoding ensures 23:00 and 01:00 are mathematically close
            X['Hour_sin'] = np.sin(2 * np.pi * X['Hour'] / 24)
            X['Hour_cos'] = np.cos(2 * np.pi * X['Hour'] / 24)
            X['Day'] = (X['Time'] / 86400).astype(int)
            logger.debug("Created time-based features")
        
        # Amount-based features
        if 'Amount' in X.columns:
            X['Amount_log'] = np.log1p(X['Amount'])
            
            # Create amount bins with NaN handling
            amount_bins = pd.cut(
                X['Amount'],
                bins=[0, 10, 50, 100, 500, float('inf')],
                labels=[0, 1, 2, 3, 4]
            )
            X['Amount_bin'] = amount_bins.fillna(0).astype(int)  # Fill NaN with 0
            
            logger.debug("Created amount-based features")
                
        # Interaction features between PCA components
        v_features = [col for col in X.columns if col.startswith('V')]
        if len(v_features) >= 2:
            X['V1_V2_interaction'] = X[v_features[0]] * X[v_features[1]]
            logger.debug("Created interaction features")
        
        self.feature_names_ = X.columns.tolist()
        logger.info(f"Feature engineering complete. Total features: {len(X.columns)}")
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names for pipeline compatibility."""
        return self.feature_names_ if self.feature_names_ is not None else input_features


def create_preprocessing_pipeline(
    scale_features: bool = True,
    engineer_features: bool = True
) -> Pipeline:
    """
    Create preprocessing pipeline for fraud detection.
    
    Args:
        scale_features: Apply StandardScaler normalization
        engineer_features: Apply feature engineering
    
    Returns:
        Sklearn Pipeline with specified transformations
    """
    steps = []
    
    if engineer_features:
        steps.append(('feature_engineering', FeatureEngineer()))
    
    if scale_features:
        steps.append(('scaler', StandardScaler()))
    
    pipeline = Pipeline(steps)
    logger.info(f"Created preprocessing pipeline: {[name for name, _ in steps]}")
    
    return pipeline
