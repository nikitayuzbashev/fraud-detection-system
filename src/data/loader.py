"""Data loading utilities for fraud detection system."""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

from ..utils.config import Config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def load_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the fraud detection dataset.
    
    Args:
        data_path: Path to the CSV file. If None, uses Config.DATA_PATH
    
    Returns:
        DataFrame with the loaded data
    
    Raises:
        FileNotFoundError: If the data file doesn't exist
    """
    path = data_path or Config.DATA_PATH
    
    if not path.exists():
        error_msg = f"Data file not found at {path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    logger.info(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target variable.
    
    Args:
        df: Input dataframe with 'Class' column as target
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    if 'Class' not in df.columns:
        raise ValueError("DataFrame must contain 'Class' column as target")
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    return X, y


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for the dataset.
    
    Args:
        df: Input dataframe
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "n_samples": len(df),
        "n_features": len(df.columns) - 1 if 'Class' in df.columns else len(df.columns),
        "n_frauds": int(df['Class'].sum()) if 'Class' in df.columns else None,
        "fraud_rate": float(df['Class'].mean()) if 'Class' in df.columns else None,
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024**2)
    }
    
    logger.info(f"Data summary: {summary}")
    
    return summary
