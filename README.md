# League of Legends Match Predictor

A machine learning project that predicts the outcome of League of Legends matches using logistic regression implemented in PyTorch.

## 📊 Project Overview

This project applies machine learning techniques to predict League of Legends match outcomes based on in-game statistics. Using a logistic regression model built with PyTorch, the system analyzes various match features to determine the probability of winning or losing.

## 🎯 Features

- **Data Preprocessing**: Automated loading, cleaning, and standardization of match data
- **Machine Learning Model**: Logistic regression implementation using PyTorch
- **Model Optimization**: L2 regularization and hyperparameter tuning
- **Performance Analysis**: Comprehensive evaluation with multiple metrics
- **Visualization**: Confusion matrices, ROC curves, and feature importance plots
- **Model Persistence**: Save and load trained models
- **Feature Importance**: Analysis of which game statistics most influence match outcomes

## 📁 Project Structure

```
lol_game_predictor/
├── Final Project League of Legends Match Predictor-v2.ipynb  # Main notebook
├── league_of_legends_data_large.csv                          # Dataset
├── README.md                                                  # This file
└── lol_predictor_model.pth                                   # Saved model (generated)
```

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab

### Required Libraries

```bash
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install torch
pip install seaborn
```

## 📈 Dataset

The project uses the `league_of_legends_data_large.csv` dataset containing match statistics including:

- Player performance metrics
- Team statistics
- Game duration and objectives
- Win/loss outcomes (target variable)

## 🚀 Usage

1. **Clone or download the project files**
2. **Open the Jupyter notebook**:
   ```bash
   jupyter notebook "Final Project League of Legends Match Predictor-v2.ipynb"
   ```
3. **Run all cells sequentially** to:
   - Load and preprocess the data
   - Train the model
   - Evaluate performance
   - Generate visualizations

## 🔬 Project Workflow

### Step 1: Data Loading and Preprocessing
- Load League of Legends dataset
- Split into training (80%) and testing (20%) sets
- Standardize features using StandardScaler
- Convert to PyTorch tensors

### Step 2: Model Implementation
- Define LogisticRegressionModel class
- Initialize model with appropriate input dimensions
- Set up Binary Cross-Entropy loss function
- Configure SGD optimizer

### Step 3: Model Training
- Train for 1000 epochs
- Monitor loss every 100 epochs
- Evaluate on both training and test sets
- Calculate accuracy metrics

### Step 4: Model Optimization
- Implement L2 regularization (weight decay)
- Retrain optimized model
- Compare performance improvements

### Step 5: Visualization and Interpretation
- Generate confusion matrix
- Plot ROC curve and calculate AUC
- Create classification report
- Analyze precision, recall, and F1-score

### Step 6: Model Persistence
- Save trained model weights
- Load and verify model performance
- Ensure consistency across sessions

### Step 7: Hyperparameter Tuning
- Test multiple learning rates [0.01, 0.05, 0.1]
- Identify optimal learning rate
- Compare performance across configurations

### Step 8: Feature Importance Analysis
- Extract model weights
- Visualize feature importance
- Identify most influential game statistics

## 📊 Model Performance

The final model achieves:
- **Training Accuracy**: ~85-90%
- **Test Accuracy**: ~85-90%
- **AUC Score**: ~0.85-0.95

*Note: Exact performance may vary based on random initialization and data splits*

## 📋 Key Insights

### Most Important Features
The model identifies key game statistics that most strongly predict match outcomes:
- Gold difference
- Kill/Death ratio
- Objective control (dragons, barons)
- Vision score
- Damage dealt

### Model Interpretability
- Positive weights indicate features that increase win probability
- Negative weights indicate features that decrease win probability
- Feature importance visualization helps understand strategic game elements

## 🔧 Customization

### Hyperparameter Tuning
Modify the following parameters in the notebook:
- `learning_rate`: Controls optimization step size
- `epochs`: Number of training iterations
- `weight_decay`: L2 regularization strength

### Feature Engineering
- Add new derived features from existing statistics
- Experiment with feature selection techniques
- Try different scaling methods

### Model Architecture
- Experiment with different activation functions
- Add regularization techniques
- Try ensemble methods

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is created for educational purposes. Dataset and game data belong to Riot Games.

## 🎮 About League of Legends

League of Legends is a multiplayer online battle arena (MOBA) game where two teams compete to destroy the opponent's base. The game generates rich statistical data that makes it ideal for machine learning applications.

## 🔍 Future Improvements

- **Advanced Models**: Implement Random Forest, XGBoost, or Neural Networks
- **Real-time Prediction**: Create API for live match prediction
- **Player-specific Models**: Personalized predictions based on player history
- **Time-series Analysis**: Incorporate match progression data
- **Feature Engineering**: Create more sophisticated derived features

## 📞 Contact

For questions or suggestions about this project, please open an issue or contact the maintainer.

---

*This project demonstrates practical application of machine learning in gaming analytics and serves as an educational resource for understanding binary classification with PyTorch.*
