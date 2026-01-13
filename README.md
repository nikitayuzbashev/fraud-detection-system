# Fraud Detection System

Production-ready fraud detection ML system using XGBoost and FastAPI.

## Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start
```bash
# Train model
python scripts/train_model.py

# Start API
uvicorn src.api.main:app --reload --port 8000
```

## Project Structure
```
fraud-detection-system/
├── src/
│   ├── data/           # Data processing
│   ├── models/         # Model training
│   ├── api/            # FastAPI application
│   └── utils/          # Utilities
├── tests/              # Test suite
├── scripts/            # Training scripts
└── data/               # Data storage
```
