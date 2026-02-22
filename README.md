# Fraud Detection System

Production-ready fraud detection system using XGBoost. Achieves 98.6% ROC-AUC and 88% recall on the Credit Card Fraud Detection dataset.

## 🎯 Key Features

- **XGBoost classifier** with SMOTE for handling severe class imbalance (0.17% fraud rate)
- **Feature engineering pipeline** with temporal features and cyclical encoding
- **Comprehensive evaluation** with precision, recall, F1-score, and ROC-AUC metrics
- **Modular architecture** with separation of concerns (data, models, utils)
- **Professional logging** with structured JSON output
- **Configuration management** via environment variables

## 📊 Performance Metrics

Trained on 284,807 credit card transactions (492 fraudulent):

| Metric | Score |
|--------|-------|
| **ROC-AUC** | **0.986** |
| **Recall** | **0.878** |
| **Precision** | 0.450 |
| **F1-Score** | 0.595 |

The model successfully catches 88% of fraudulent transactions while maintaining an acceptable false positive rate.

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Virtual environment

### Installation
```bash
# Clone repository
git clone https://github.com/nikitayuzbashev/fraud-detection-system.git
cd fraud-detection-system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Get Dataset

Download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/whenamancodes/fraud-detection) from Kaggle and place `creditcard.csv` in `data/raw/`.

### Train Model
```bash
python scripts/train_model.py
```

The trained model will be saved to `data/models/fraud_detector.joblib`.

## 📁 Project Structure
```
fraud-detection-system/
├── src/
│   ├── data/
│   │   ├── loader.py          # Data loading utilities
│   │   └── preprocessor.py    # Feature engineering pipeline
│   ├── models/
│   │   ├── trainer.py         # XGBoost model training
│   │   └── evaluator.py       # Model evaluation metrics
│   └── utils/
│       ├── config.py          # Configuration management
│       └── logger.py          # Logging setup
├── scripts/
│   └── train_model.py         # Training script
├── data/
│   ├── raw/                   # Raw datasets
│   ├── processed/             # Processed data
│   └── models/                # Trained models
└── requirements.txt
```

## 🛠️ Technical Stack

- **ML Framework:** XGBoost, scikit-learn, imbalanced-learn
- **Data Processing:** pandas, numpy
- **Configuration:** python-dotenv
- **Logging:** python-json-logger

## 💡 Key Implementation Details

### Handling Class Imbalance

The dataset has severe class imbalance (0.17% fraud). We address this using:
- **SMOTE** (Synthetic Minority Over-sampling Technique) for training
- **Stratified train-test split** to maintain fraud rate in both sets
- **Threshold tuning** optimized for F1-score

### Feature Engineering

- **Temporal features:** Hour of day with cyclical encoding (sin/cos)
- **Amount features:** Log transformation and binning
- **Interaction features:** Cross-products of PCA components

### Pipeline Design

All preprocessing steps are bundled in a scikit-learn pipeline:
```
Feature Engineering → Scaling → SMOTE → XGBoost
```

This ensures consistent transformations during training and inference.

## 📈 Model Performance Analysis

The model prioritises **recall over precision**, which is appropriate for fraud detection:
- **High Recall (88%):** Catches most fraud cases, minimising financial losses
- **Moderate Precision (45%):** Some false alarms, but manual review is acceptable
- **Excellent ROC-AUC (98.6%):** Near-perfect discrimination capability

This tradeoff is optimal for fraud detection, where missing fraud is more costly than false alarms.

## 🔧 Configuration

Key settings in `.env`:
```bash
# Model Configuration
N_ESTIMATORS=100
MAX_DEPTH=6
LEARNING_RATE=0.1
RANDOM_STATE=42

# Data Configuration
DATA_PATH=data/raw/creditcard.csv
MODEL_PATH=data/models/fraud_detector.joblib
```

## 📝 License

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/whenamancodes/fraud-detection) from Kaggle.

## 👤 Author

Nikita Yuzbashev
- GitHub: [@nikitayuzbashev](https://github.com/nikitayuzbashev)
