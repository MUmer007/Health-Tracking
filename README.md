#  MediTrack Health-Tracking System

> AI-powered clinical risk assessment platform for predictive healthcare screening

[![Live Demo](<img width="1742" height="1248" alt="screencapture-healthtracking-jc9kdrfdjvfxmeedwvv6bl-streamlit-app-2025-11-29-01_15_31" src="https://github.com/user-attachments/assets/3b15ce86-988d-4ccb-acea-8b40ba1f4a52" />)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[🚀 Live Demo](https://healthtracking-jc9kdrfdjvfxmeedwvv6bl.streamlit.app/) | [🐛 Report Bug](https://github.com/yourusername/meditrack/issues) | [✨ Request Feature](https://github.com/yourusername/meditrack/issues)

![MediTrack Demo](https://raw.githubusercontent.com/yourusername/meditrack/main/static/demo-screenshot.png)

---

## 📖 Overview

MediTrack is a production-ready machine learning application that provides real-time risk assessment for four major health conditions: Type 2 Diabetes, Breast Cancer, Heart Disease, and Kidney Disease. Built with scikit-learn and deployed on Streamlit, the system uses calibrated probability models to deliver clinical-grade predictions.

**⚠️ This is an educational tool and should not be used for medical diagnosis.** All predictions require professional medical interpretation.

## ✨ Key Features

- 🔬 Multi-disease risk screening with probability calibration
- ⚡ Real-time inference with sub-second latency
- 🔒 Privacy-preserving architecture (zero data persistence)
- 🎨 Responsive UI with clinical design standards
- 📦 Production-ready model artifacts with versioning

## 📊 Performance Metrics

Models evaluated on held-out test sets with stratified sampling:

| Disease | AUROC | AUPRC | Brier Score | Algorithm |
|---------|-------|-------|-------------|-----------|
| Breast Cancer | 0.964 | 0.946 | 0.075 | Gradient Boosting |
| Kidney Disease | 1.000 | 1.000 | 0.028 | Gradient Boosting |
| Diabetes | 0.823 | 0.702 | 0.167 | Logistic Regression |
| Heart Disease | 0.783 | 0.837 | 0.207 | Gradient Boosting |

All models exceed the clinical threshold of 0.75 AUROC for medical ML applications.

## 🏗️ Architecture

### ML Pipeline

```
Input Layer (6 features)
    ↓
Feature Engineering (DictVectorizer)
    ↓
Base Model (LogisticRegression / GradientBoosting)
    ↓
Probability Calibration (Platt Scaling)
    ↓
Conformal Prediction (90% confidence)
    ↓
Risk Stratification (3-tier)
```

### Training Pipeline

1. **Data Split**: 65% train / 15% calibration / 20% test (stratified)
2. **Hyperparameter Tuning**: RandomizedSearchCV with 5-fold CV
3. **Model Selection**: Best performer by AUROC
4. **Calibration**: Sigmoid scaling on calibration set
5. **Validation**: Independent test set evaluation
6. **Serialization**: Joblib for model persistence

### 🛠️ Tech Stack

- **Framework**: Streamlit 1.28+
- **ML**: scikit-learn 1.3+, NumPy, Pandas
- **Models**: Logistic Regression, Gradient Boosting Classifier
- **Calibration**: CalibratedClassifierCV (Platt scaling)
- **Deployment**: Streamlit Cloud / Docker

## 💻 Installation

### Requirements

```bash
Python 3.8+
pip or conda
```

### Setup

```bash
git clone https://github.com/yourusername/meditrack-health.git
cd meditrack-health
pip install -r requirements.txt
streamlit run app.py
```

Access at `http://localhost:8501`

### 🐳 Docker

```bash
docker build -t meditrack:latest .
docker run -p 8501:8501 meditrack:latest
```

## 📝 Usage

### Input Features

| Feature | Type | Range | Unit |
|---------|------|-------|------|
| Age | int | 1-120 | years |
| BMI | float | 10.0-50.0 | kg/m² |
| Glucose | float | 40-350 | mg/dL |
| Blood Pressure | float | 40-220 | mmHg |
| Heart Rate | int | 30-240 | bpm |
| Temperature | float | 94-105 | °F |

### 🎯 Risk Classification

- **🟢 Low Risk** (p < 0.4): Routine monitoring recommended
- **🟡 Moderate Risk** (0.4 ≤ p < 0.7): Clinical consultation advised
- **🔴 High Risk** (p ≥ 0.7): Immediate medical evaluation required

Thresholds optimized for clinical sensitivity/specificity balance.

## 🧪 Model Training

Training pipeline implemented in `model_training.ipynb`:

### Hyperparameter Search Space

**Logistic Regression:**
```python
{
    'C': np.logspace(-3, 2, 15),
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear'],
    'class_weight': ['balanced']
}
```

**Gradient Boosting:**
```python
{
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.03, 0.1, 0.2],
    'max_depth': [2, 3, 4],
    'subsample': [0.7, 0.85, 1.0]
}
```

### 📈 Evaluation Metrics

- **AUROC**: Overall discrimination capability
- **AUPRC**: Precision-recall tradeoff (critical for imbalanced data)
- **Brier Score**: Calibration quality (lower is better)
- **Conformal Intervals**: Uncertainty quantification at 90% confidence

### 🔄 Reproducibility

All experiments use `RANDOM_SEED=42` for deterministic results. Model artifacts versioned with training timestamps.

## 📁 Project Structure

```
meditrack-health/
├── app.py                              # Streamlit application
├── model_training.ipynb                # Training pipeline
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Container config
│
├── artifacts/
│   ├── meta.json                       # Task configuration
│   ├── tuning_results.json             # Hyperparameter results
│   ├── {disease}_train.csv             # Training data
│   ├── {disease}_calib.csv             # Calibration data
│   ├── {disease}_test.csv              # Test data
│   ├── {disease}_vectorizer.joblib     # Feature transformer
│   ├── {disease}_calibrated_model.joblib   # Production model
│   └── {disease}_conformal.json        # Confidence thresholds
│
└── static/
    └── Blue-Modern-Medical-Care-Logo.jpg
```

## 🚀 Deployment

### ☁️ Streamlit Cloud

1. Fork repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy from `main` branch
4. Configure secrets if needed

### ⚙️ Production Considerations

- **📊 Monitoring**: Track prediction latency and error rates
- **🔖 Model Versioning**: Use semantic versioning (e.g., `v1.2.0`)
- **🧪 A/B Testing**: Shadow mode for new models before promotion
- **🔄 Fallback**: Handle model loading failures gracefully
- **📝 Logging**: Structured logs for debugging (no PHI)

### Environment Variables

```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
```

## 🔌 API Integration (Future)

Planned RESTful endpoint structure:

```http
POST /api/v1/predict
Content-Type: application/json

{
  "disease": "diabetes",
  "features": {
    "Age": 45,
    "BMI": 28.5,
    "Glucose": 140,
    "BP": 130,
    "HR": 75,
    "Temp": 98.6
  }
}
```

Response:
```json
{
  "probability": 0.65,
  "risk_level": "moderate",
  "confidence_interval": [0.55, 0.75],
  "model_version": "v1.0.0"
}
```

## 🤝 Contributing

### 🔧 Development Setup

```bash
# Clone and install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black .
isort .

# Linting
flake8 app.py
```

### 📋 Pull Request Process

1. Fork the repo and create a feature branch
2. Add tests for new functionality
3. Ensure all tests pass
4. Update documentation as needed
5. Submit PR with clear description

### 📐 Code Style

- Follow PEP 8 conventions
- Use type hints where applicable
- Document functions with docstrings
- Keep functions focused and modular

## ⚠️ Known Limitations

- **Feature Space**: Limited to 6 input parameters
- **Generalization**: Trained on specific datasets; may not generalize to all populations
- **Temporal Drift**: Model performance may degrade over time
- **Clinical Context**: Predictions require professional interpretation
- **Liability**: Not FDA-approved; educational use only

## 🔐 Security & Privacy

- **🚫 No Data Storage**: Patient inputs never persisted to disk or database
- **🔄 Stateless Design**: Each prediction is independent
- **🔓 No Authentication**: Privacy by design (no user tracking)
- **🔒 HTTPS**: Encrypted transmission via Streamlit Cloud
- **📋 Audit Trail**: System logs only (no PHI)

This application is **HIPAA-aware** but not HIPAA-compliant (no PHI storage). Not intended for production clinical use without proper validation and regulatory approval.

## 🗺️ Roadmap

### v1.1 (Q1 2026)
- 📄 Export predictions to PDF
- 🔬 Additional disease models (COPD, Stroke)
- 📊 Model explainability (SHAP values)

### v1.2 (Q2 2026)
- 🔌 RESTful API endpoints
- 📈 Historical tracking dashboard
- 🌍 Multi-language support

```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 📖 References

1. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR.
2. Zadrozny & Elkan (2002). Transforming Classifier Scores into Accurate Multiclass Probability Estimates. KDD.
3. Vovk et al. (2005). Algorithmic Learning in a Random World. Springer.

## 📞 Contact

- **🐛 Issues**: [GitHub Issues](https://github.com/dashboard)
- **📧 Email**: uneebashaikh33@gmail.com


---

**Maintained by the MediTrack Development Team** | Last Updated: November 2025
 
