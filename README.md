# 📝 Breast Cancer Classifier Project

## Project Overview
This project focuses on building machine learning models to classify breast cancer as **malignant** or **benign** using the **Breast Cancer Wisconsin (Diagnostic) Dataset**. It explores both traditional ML algorithms and deep learning approaches. The project evaluates the performance of models like **K-Nearest Neighbors (KNN)**, **Naive Bayes**, and **Deep Neural Networks (DNN)** with varying layers, comparing their performance on various metrics.

---

## 🔧 Required Libraries

Before running the project, ensure you have the following libraries installed:

```bash
pip install pandas scikit-learn matplotlib seaborn tensorflow gdown
```

---

## 📂 Dataset

The dataset used in this project is the **Breast Cancer Wisconsin (Diagnostic) Data Set**
---

## 🛠️ Data Preparation and Cleaning

1. **Removing Unnecessary Columns:** The `id` and `Unnamed: 32` columns are dropped since they are irrelevant to the classification task.
2. **Mapping Diagnosis to Numerical Values:** The `diagnosis` column is mapped to `1` for malignant tumors and `0` for benign tumors.

---

## 📊 Data Visualization

The dataset is visualized to gain insights into its distribution and key features, including:

- Distribution of diagnosis counts (Benign vs. Malignant)
- Correlation matrix to show feature relationships
- Boxplots and histograms for feature analysis

---

## 🤖 Machine Learning Models

### 1. K-Nearest Neighbors (KNN)
We implemented a KNN model to classify tumors as benign or malignant. The performance of this model was evaluated using metrics such as **Accuracy**, **Precision**, **Recall**, **F1-Score**, and the **ROC-AUC Curve**.

### 2. Naive Bayes Classifier
A simple **Naive Bayes** model was trained on the same dataset, and its performance was compared with KNN.

---

## 🧠 Deep Learning Model

### DNN Architectures
Three different architectures were built using TensorFlow/Keras:

- **1-Layer DNN**
- **3-Layer DNN**
- **5-Layer DNN**

Each model was trained and evaluated for **accuracy**, and the models' performances were compared using confusion matrices and classification reports.

---

## ⚖️ Model Performance Comparison

The performance of the **KNN** model and the **3-Layer DNN** model was compared using the following metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **AUC (Area Under the Curve)**

Visualizations were generated to illustrate the model performances and make a clear comparison between traditional ML and DNN.

---

## 📈 Results Summary

- **KNN** achieved an accuracy of **97.66%** with a precision of **98.36%**.
- **3-Layer DNN** outperformed KNN with an accuracy of **99.41%** and perfect precision for the benign class.

---

## 📊 Evaluation Metrics Visualization

Bar charts were plotted to compare the evaluation metrics for **KNN** and the **3-Layer DNN** model, providing a clear understanding of their performances.

---

## 📅 Future Enhancements

- **Additional Model Exploration:** Try out advanced architectures like **CNN** or **RNN** for better feature extraction.
- **Hyperparameter Tuning:** Perform extensive hyperparameter optimization to improve model performance.
- **Ensemble Learning:** Combine multiple classifiers to create a stronger model using ensemble techniques.
