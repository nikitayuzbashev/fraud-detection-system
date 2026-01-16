"""Configuration management for the fraud detection system."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""
    
    # Project paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = DATA_DIR / "models"
    
    # Application
    APP_NAME: str = os.getenv("APP_NAME", "fraud-detection-api")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # Model Configuration
    MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", str(MODELS_DIR / "fraud_detector.joblib")))
    THRESHOLD: float = float(os.getenv("THRESHOLD", "0.5"))
    
    # Data Configuration
    DATA_PATH: Path = Path(os.getenv("DATA_PATH", str(RAW_DATA_DIR / "creditcard.csv")))
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")
    
    # Model Training
    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))
    TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.2"))
    N_ESTIMATORS: int = int(os.getenv("N_ESTIMATORS", "100"))
    MAX_DEPTH: int = int(os.getenv("MAX_DEPTH", "6"))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "0.1"))
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, cls.MODELS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production environment."""
        return cls.ENVIRONMENT.lower() == "production"
    
    @classmethod
    def get_model_config(cls) -> dict:
        """Get model training configuration as a dictionary."""
        return {
            "n_estimators": cls.N_ESTIMATORS,
            "max_depth": cls.MAX_DEPTH,
            "learning_rate": cls.LEARNING_RATE,
            "random_state": cls.RANDOM_STATE,
            "eval_metric": "logloss",
            "use_label_encoder": False,
        }


# Create directories on import
Config.create_directories()
