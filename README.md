# Detecting Patterns in Tabular Medical Data with MIMIC-III

This project focuses on analyzing healthcare data from the MIMIC-III dataset using classical machine learning models and deep learning techniques. The goal is to derive meaningful insights from patient data by applying various algorithms, including supervised and unsupervised learning methods, and to explore the use of neural networks in predictive modeling.

## Project Overview

- **Dataset**: MIMIC-III, a publicly available critical care database.
- **Tools and Libraries**: Python, NumPy, Pandas, Matplotlib, scikit-learn, TensorFlow (Keras).
- **Techniques Used**:
  - Supervised Learning: Linear Regression, Decision Trees, Support Vector Machines (SVM), k-Nearest Neighbors (kNN), Logistic Regression.
  - Unsupervised Learning: K-means Clustering, Principal Component Analysis (PCA), t-Distributed Stochastic Neighbor Embedding (t-SNE).
  - Deep Learning: Deep Neural Networks (DNN) for classification tasks.

## Key Components

1. **Classical Machine Learning:**
   - Implemented various supervised learning algorithms to predict outcomes based on features such as age, BMI, and blood sodium.
   - Explored regression models to predict continuous variables.
   - Evaluated model performance using metrics like accuracy, RMSE, MSE, R-squared, and visualized results with graphs and plots.

2. **Unsupervised Learning:**
   - Applied dimensionality reduction techniques (PCA, t-SNE) to visualize data in reduced dimensions.
   - Used K-means clustering to group patients based on multiple health parameters and assessed clustering quality with Silhouette and Davies-Bouldin scores.

3. **Deep Learning:**
   - Developed a Deep Neural Network (DNN) for binary classification tasks.
   - Experimented with different model architectures, activation functions, optimizers, and evaluated performance using training/validation accuracy and loss graphs.

## How to Use

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YourUsername/YourRepositoryName.git
