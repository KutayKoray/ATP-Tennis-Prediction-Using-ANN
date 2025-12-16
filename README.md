# 🎾 Tennis Match Prediction with Neural Networks from Scratch

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-From%20Scratch-green.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

> **A journey from zero to beating IBM's AI predictions at Wimbledon 2025**

This project implements a complete neural network system **from scratch** (no TensorFlow, no PyTorch) to predict professional tennis match outcomes. The ultimate goal: **outperform IBM's SlamTracker AI** in predicting Wimbledon 2025 tournament results.

## 🎯 Project Mission

**Build a neural network from the ground up and prove it can compete with commercial AI systems.**

Starting with nothing but NumPy and raw ATP tennis data, this project:
1. ✅ Implements a 2-hidden-layer neural network from scratch
2. ✅ Engineers 37 advanced features from historical match data
3. ✅ Achieves 67.4% accuracy on 2024 test matches
4. ✅ **Beats IBM's Slamtracker by 1.6% on Wimbledon 2025 predictions** (66.1% vs 64.6%)

## 🏆 Key Achievements

| Metric | Result |
|--------|--------|
| **Test Accuracy (2024)** | 67.4% |
| **Wimbledon 2025 Accuracy** | 66.1% (84/127 matches) |
| **IBM's Slamtracker Accuracy** | 64.6% (82/127 matches) |
| **Improvement over IBM** | +1.6% (+2 matches) |
| **Training Data** | 150,000+ ATP matches (1968-2024) |
| **Features Engineered** | 37 advanced features |

### 📊 Round-by-Round Performance vs IBM

| Round | Our Model | IBM's Slamtracker | Difference |
|-------|-----------|-------------------|------------|
| Round 1 | 59.4% (38/64) | 56.2% (36/64) | **+3.1%** ✅ |
| Round 2 | 65.6% (21/32) | 68.8% (22/32) | -3.1% |
| Round 3 | 75.0% (12/16) | 56.2% (9/16) | **+18.8%** ✅ |
| Round 4 | 100.0% (8/8) | 100.0% (8/8) | Tied |
| Quarter Finals | 50.0% (2/4) | 100.0% (4/4) | -50.0% |
| Semi Finals | 100.0% (2/2) | 100.0% (2/2) | Tied |
| Final | 100.0% (1/1) | 100.0% (1/1) | Tied |

**Notable Achievement**: Perfect prediction of the Wimbledon 2025 Final (Sinner vs Alcaraz) with 99% confidence!

## 🧠 Technical Highlights

### Neural Network Architecture (Built from Scratch)
```
Input Layer: 37 features
    ↓
Hidden Layer 1: 128 neurons (Leaky ReLU, α=0.01)
    ↓
Hidden Layer 2: 64 neurons (Leaky ReLU, α=0.01)
    ↓
Output Layer: 1 neuron (Sigmoid)
```

### Advanced Features
- **No Deep Learning Frameworks**: Pure NumPy implementation
- **Custom Backpropagation**: Manually implemented gradient descent
- **ADAM Optimizer**: From-scratch implementation with momentum
- **L2 Regularization**: Prevents overfitting (λ=0.3)
- **He Initialization**: Optimal weight initialization for ReLU variants
- **Early Stopping**: Validation-based training termination
- **StandardScaler Normalization**: Critical for L2 regularization

### Feature Engineering Pipeline
- **37 Engineered Features** including:
  - Historical win rates (career, surface-specific, recent form)
  - Service statistics (ace rates, double fault rates, first serve %)
  - Head-to-head records
  - Tournament context (importance, surface type)
  - Player attributes (age, height, experience, handedness)
  - Temporal features (cyclical month encoding)
  - Interaction features (rank×surface, form differences)

### Data Leakage Prevention
- ✅ Chronological feature computation
- ✅ No in-match statistics used
- ✅ Historical averages from past matches only
- ✅ Strict temporal train/validation/test split

## 📁 Project Structure

```
ANN/
|
├── requirements.txt                   # Python dependencies
│
├── datas/                             # Raw ATP match data
│   ├── atp_matches_1968.csv          # Historical matches by year
│   ├── atp_matches_1969.csv
│   ├── ...
│   ├── atp_matches_2024.csv
│   └── README.md                      # Data source documentation
│
├── notebooks/                         # Main project notebooks
│   ├── 2HLayer_ANN.ipynb             # 🎯 Main neural network implementation
│   │
│   ├── FeatureEngineering/           # Feature engineering pipeline
│   │   ├── 01_data_inspection.py     # Exploratory data analysis
│   │   ├── 02_data_cleaning.py       # Data cleaning & preprocessing
│   │   ├── 03_feature_engineering.py # Feature creation (37 features)
│   │   ├── 04_create_npz_dataset.py  # Final dataset preparation
│   │   ├── 05_correlation_analysis.py # Feature correlation analysis
│   │   ├── run_all.py                # Run entire pipeline
│   │   └── atp_featured_dataset.npz  # Final training dataset
│   │
│   └── 2025Wimbledon/                # Wimbledon 2025 predictions
│       ├── prepare_wimbledon_features.py
│       ├── normalize_wimbledon.py
│       ├── 2025_wimbledon_matches.csv
│       ├── IBM's_predictions.csv     # IBM SlamTacker predictions
│       └── rounds/                   # Round-by-round results
│           ├── Round1_FirstRound.csv
│           ├── Round2_SecondRound.csv
│           └── ...
│
└── README.md                          # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12+ (tested on 3.12.7)
- 8GB+ RAM recommended
- macOS, Linux, or Windows

### Installation

1. **Clone the repository**
```bash
cd ~/Desktop
git clone <your-repo-url> ANN
cd ANN
```

2. **Create virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Quick Start

#### Option 1: Run Pre-trained Model (Fastest)

```bash
# Open Jupyter notebook
jupyter notebook notebooks/2HLayer_ANN.ipynb

# In the notebook:
# 1. Run cells 1-3 to load the pre-trained dataset
# 2. Skip to cell 11 to load saved model parameters (if available)
# 3. Run evaluation cells to see results
```

#### Option 2: Train from Scratch (Full Experience)

```bash
# Step 1: Prepare the dataset
cd notebooks/FeatureEngineering
python run_all.py  # Runs entire feature engineering pipeline (~10-15 minutes)

# Step 2: Train the model
cd ..
jupyter notebook 2HLayer_ANN.ipynb

# In the notebook:
# 1. Run all cells in sequence
# 2. Choose Phase 1 (find best epoch with validation)
# 3. Choose Model 2 (Leaky ReLU + He initialization - recommended)
# 4. Wait for training (~5-10 minutes)
# 5. Note the best epoch number
# 6. Run Phase 2 with the best epoch for final training
```

## 📖 Detailed Usage Guide

### 1. Feature Engineering Pipeline

The feature engineering process transforms raw ATP match data into ML-ready features:

```bash
cd notebooks/FeatureEngineering

# Option A: Run entire pipeline
python run_all.py

# Option B: Run step-by-step
python 01_data_inspection.py      # Analyze raw data
python 02_data_cleaning.py        # Clean and preprocess
python 03_feature_engineering.py  # Create 37 features
python 04_create_npz_dataset.py   # Prepare final dataset
```

**Output**: `atp_featured_dataset.npz` (37 features × 150,000+ matches)

### 2. Training the Neural Network

Open `notebooks/2HLayer_ANN.ipynb` in Jupyter:

#### Phase 1: Hyperparameter Tuning
```python
# Cell 1: Load data
# Cell 2: Configure hyperparameters
# Cell 3-4: Define model functions
# Cell 5: Training control

# When prompted:
# - Choose option 1 (Phase 1 - Find best epoch)
# - Choose option 2 (Leaky ReLU + He initialization)

# Training will run with early stopping
# Note the "best epoch" number (e.g., 1350)
```

#### Phase 2: Final Training
```python
# Run the training cell again:
# - Choose option 2 (Phase 2 - Final training)
# - Enter the best epoch number from Phase 1
# - Model trains on full dataset (train + validation)
```

#### Evaluation
```python
# Run evaluation cells to see:
# - Test set accuracy (2024 matches)
# - Confidence distribution analysis
# - Detailed match predictions
```

### 3. Wimbledon 2025 Predictions

```bash
cd notebooks/2025Wimbledon

# Prepare Wimbledon data
python prepare_wimbledon_features.py
python normalize_wimbledon.py

# Run predictions (in Jupyter notebook)
# Execute the Wimbledon prediction cell in 2HLayer_ANN.ipynb
```

### 4. Comparing with IBM SlamTacker

The notebook automatically compares predictions with IBM's:
- Round-by-round accuracy comparison
- Confidence level analysis
- Head-to-head performance metrics

Results are saved in `notebooks/win.txt`

## 🔧 Configuration & Hyperparameters

### Model Architecture
```python
N_X = 37          # Input features
N_H1 = 128        # First hidden layer
N_H2 = 64         # Second hidden layer
N_Y = 1           # Output (binary classification)
```

### Training Parameters
```python
LEARNING_RATE = 0.003    # ADAM learning rate
EPOCHS = 5000            # Maximum epochs
PATIENCE = 10            # Early stopping patience
LAMBDA = 0.3             # L2 regularization strength
BETA1 = 0.9              # ADAM momentum
BETA2 = 0.999            # ADAM RMSprop
```

### Recommended Settings
- **Activation**: Leaky ReLU (α=0.01) for hidden layers
- **Initialization**: He initialization
- **Optimizer**: ADAM (faster convergence than SGD)
- **Normalization**: StandardScaler (mean=0, std=1)

## 📊 Understanding the Results

### Accuracy Metrics

**Overall Test Accuracy**: 67.4% on 2024 matches
- Better than random (50%)
- Better than "always pick higher rank"
- Better than logistic regression
- Competitive with commercial systems

### Feature Importance

Top predictive features:
1. Recent form (last 10 matches)
2. Surface-specific win rate
3. Head-to-head record
4. Historical service statistics
5. Ranking points differential

## 🎓 Learning Outcomes

This project demonstrates:

### Deep Learning Fundamentals
- ✅ Forward propagation implementation
- ✅ Backpropagation and gradient computation
- ✅ Activation functions (Sigmoid, Tanh, Leaky ReLU)
- ✅ Weight initialization strategies (Xavier, He)
- ✅ Optimization algorithms (ADAM)
- ✅ Regularization techniques (L2, early stopping)

### Machine Learning Best Practices
- ✅ Data preprocessing and normalization
- ✅ Feature engineering and selection
- ✅ Train/validation/test splitting
- ✅ Hyperparameter tuning
- ✅ Model evaluation and validation
- ✅ Data leakage prevention

### Software Engineering
- ✅ Modular code architecture
- ✅ Comprehensive documentation
- ✅ Version control and reproducibility
- ✅ Performance optimization

## 🐛 Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: atp_featured_dataset.npz`
```bash
# Solution: Run feature engineering pipeline first
cd notebooks/FeatureEngineering
python run_all.py
```

**Issue**: Training accuracy stuck at 50%
```bash
# Solution: Check normalization
# Ensure StandardScaler is applied (not max normalization)
# Verify in 04_create_npz_dataset.py
```

**Issue**: Model overfitting (train acc >> test acc)
```bash
# Solution: Increase L2 regularization
LAMBDA = 0.5  # Try higher values (0.3 → 0.5 → 0.7)
```

**Issue**: Training too slow
```bash
# Solution: Reduce dataset size or epochs
EPOCHS = 2000  # Reduce from 5000
# Or use smaller subset of data for testing
```

**Issue**: Memory error during feature engineering
```bash
# Solution: Process data in chunks
# Modify 03_feature_engineering.py to process years separately
```

## 📈 Performance Optimization Tips

### Faster Training
1. Use Leaky ReLU (faster than Tanh)
2. Use ADAM optimizer (faster than SGD)
3. Reduce PATIENCE for quicker early stopping
4. Use smaller validation set

### Better Accuracy
1. Increase hidden layer sizes (128→256, 64→128)
2. Add more features (player injury history, weather data)
3. Tune learning rate (try 0.001, 0.003, 0.01)
4. Experiment with L2 lambda (0.1, 0.3, 0.5)
5. Use ensemble methods (train multiple models)

## 🔬 Experimental Features

### Try These Modifications

**Add a third hidden layer**:
```python
N_H3 = 32  # Add between H2 and output
# Update forward/backward propagation accordingly
```

**Try different activation functions**:
```python
# ReLU
A = np.maximum(0, Z)

# ELU (smoother than ReLU)
A = np.where(Z > 0, Z, alpha * (np.exp(Z) - 1))
```

**Implement dropout regularization**:
```python
# During training
keep_prob = 0.8
mask = np.random.rand(*A.shape) < keep_prob
A = A * mask / keep_prob
```

## 🤝 Contributing

This is an educational project, but improvements are welcome!

### Areas for Contribution
- Additional feature engineering ideas
- Alternative neural network architectures
- Performance optimizations
- Better visualization tools
- Extended documentation

## 🙏 Acknowledgments

- **Jeff Sackmann**: For maintaining the tennis_atp dataset

## 📧 Contact

For questions, suggestions, or collaboration:
- Open an issue on GitHub


## 🎯 Final Thoughts

This project proves that **understanding fundamentals beats using black-box libraries**. By building a neural network from scratch, we:
- Gained deep understanding of how neural networks work
- Achieved competitive performance with commercial AI systems
- Demonstrated that careful feature engineering matters more than complex architectures

**Key Takeaway**: You don't need TensorFlow or PyTorch to build effective AI systems. Understanding the math and implementing it yourself leads to better insights and more robust solutions.

---

**⭐ If you found this project helpful, please star the repository!**

**🎾 Happy Predicting!**

---
