# diabetes_prediction-using_pytorch_framework
This project presents an end-to-end machine learning pipeline designed to predict disease progression using structured clinical data. Built on the diabetes dataset, it integrates comprehensive exploratory data analysis (EDA), robust data preprocessing, feature engineering, and deep learning modeling. The workflow begins with loading and transforming the dataset into a structured format, followed by systematic handling of missing values through imputation and removal strategies. Outliers are detected using both Interquartile Range (IQR) and Z-score methods, ensuring cleaner and more reliable input data. All features are converted into numeric form to maintain consistency across the modeling pipeline.

To preserve the temporal nature of the dataset and avoid data leakage, the project employs a TimeSeriesSplit validation strategy. This ensures that training and validation sets follow a realistic chronological order, making the evaluation more robust and production-relevant. The processed data is then converted into PyTorch tensors and organized using DataLoader for efficient batch processing during training.

Three different models are implemented and compared: a Multi-Layer Perceptron (MLP), a Linear Regression model implemented as a single-layer neural network, and a Deep Neural Network (DNN). Each model is trained using the Adam optimizer with a learning rate of 0.01 over 100 epochs, optimizing primarily for Mean Squared Error (MSE). The training process includes tracking both training and validation loss across epochs, allowing clear visibility into model convergence and potential overfitting. Additionally, PyTorch forward hooks are used to analyze activation distributions across layers, offering deeper interpretability into how the neural network processes data internally.

## 🔷 Key Highlights
-  Advanced EDA (pairplots, heatmaps, distributions)
-  Data preprocessing (missing values, outliers, normalization)
-  Feature engineering (polynomial features, correlation filtering)
-  TimeSeries-aware validation using TimeSeriesSplit
-  Multiple models:
1. MLP (Multi-Layer Perceptron)
2. Linear Regression (Neural equivalent)
3. Deep Neural Network (DNN)
-  Loss tracking (Train vs Validation)
-  Best model selection based on validation loss
-  Prediction vs Actual visualization

## Exploratory Data Analysis (EDA)
- Pairplots → feature relationships
- Heatmap → correlation analysis
- Histograms → feature distributions
- Boxplots → outlier detection
- Scatter plots → feature vs target

## 🔷 Model Architecture
1. MLP
- Input → 64 → 32 → Output
-Activation: ReLU

2. Linear Regression
- Single Dense Layer
- No activation
  
3. DNN
- Input → 64 → 32 → Output
- Deeper representation learning

## Validation Strategy
- TimeSeriesSplit (5 splits)
- Preserves temporal order
- Prevents data leakage

## Model Evaluation
1. Training Loss vs Validation Loss curves
2. Metrics used:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Best model selected based on:
3. Lowest average validation loss

## Tech Stack
Python
PyTorch
Scikit-learn
Pandas, NumPy
Matplotlib, Seaborn

The results demonstrate strong predictive capability, with the best model effectively capturing the underlying trends in disease progression. This project not only highlights the application of deep learning in healthcare analytics but also emphasizes best practices in data preprocessing, feature selection, and model evaluation.

## Demo:
https://colab.research.google.com/drive/1aOcSa-wOCxRCWoN9vP1d8RlxLMZdECm4#scrollTo=X1fu57pSk8Jm
