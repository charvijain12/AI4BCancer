# AI4BCancer: Breast Cancer Classification 🩺

Welcome to AI4BCancer, a dedicated repository focusing on leveraging deep learning techniques for the classification of breast cancer. This project aims to provide advanced and accurate models to assist in the diagnosis and early detection of breast cancer using state-of-the-art deep learning architectures.

## Table of Contents 📚

- [Introduction](#Introduction)
- [Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- [Deep Learning Models](https://github.com/footcricket05/BCancerAI/tree/main/Models)
  - [Models with PCA](https://github.com/footcricket05/BCancerAI/tree/main/Models/Models%20with%20PCA)
- [Usage](#usage)
- [Contributing](https://github.com/footcricket05/BCancerAI/blob/main/CONTRIBUTING.md)
- [License](https://github.com/footcricket05/BCancerAI/blob/main/LICENSE)

## Introduction 🌟

Breast cancer is a significant global health issue, and early detection is crucial for effective treatment. AI4BCancer focuses on utilizing cutting-edge deep learning architectures to analyze and classify breast cancer, offering a powerful tool for medical professionals.

## Dataset 📊

The dataset used for training and evaluation is the [Diagnostic Wisconsin Breast Cancer Database](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data), a comprehensive collection of breast cancer images with corresponding labels. This dataset ensures a diverse and representative set of cases for robust deep learning model training.

## AI/ML Models 🧠

Our repository specializes in various AI/ML models, each meticulously crafted for optimal performance in breast cancer classification. These models cover a spectrum of advanced techniques, including:

- **ANN**
  - [ANN Model Directory](https://github.com/footcricket05/BCancerAI/tree/main/Models/ANN)
  - The ANN model is a powerful deep learning architecture, achieving an outstanding accuracy of 99.12%. Its multi-layered structure enables it to learn complex patterns, making it a top-performer in breast cancer classification.
    
- **AutoML Model**
  - [AutoML Model Directory](https://github.com/footcricket05/BCancerAI/tree/main/Models/AutoML%20Model)
  - The AutoML model showcases its efficiency by automatically selecting the best machine learning pipeline for breast cancer classification, resulting in an impressive accuracy of 98.25%.

- **CNN Model**
  - [CNN Model Directory](https://github.com/charvijain12/AI4BCancer/tree/main/Models/CNN)
  - An efficient Convolutional Neural Network (CNN) model achieved a notable accuracy of 94.74% in classifying breast cancer, demonstrating its efficacy in automated machine learning pipelines.
  
- **CatBoost**
  - [CatBoost Model Directory](https://github.com/footcricket05/BCancerAI/tree/main/Models/CatBoost)
  - CatBoost, a gradient boosting algorithm, demonstrates high accuracy (97.37%) in breast cancer classification. Its ability to handle categorical features and robust training makes it a reliable choice.

- **Decision Tree**
  - [Decision Tree Model Directory](https://github.com/footcricket05/BCancerAI/tree/main/Models/Decision%20Tree)
  - The Decision Tree model achieves a commendable accuracy of 94.74%. It makes decisions based on hierarchical tree-like structures, providing interpretable results for breast cancer diagnosis.

- **Ensemble Method**
  - [Ensemble Method Model Directory](https://github.com/footcricket05/BCancerAI/tree/main/Models/Ensemble%20Method)
  - The Ensemble Model combines the strengths of multiple base models, resulting in a high accuracy of 96.49%. This collaborative learning approach enhances overall predictive performance.
  
- **GMM**
  - [GMM Model Directory](https://github.com/footcricket05/BCancerAI/tree/main/Models/GMM%20Model)
  - GMM exhibits a limited accuracy of 7.02%. While it may not be suitable for this breast cancer classification task, GMM is commonly used for density estimation and clustering in other scenarios.
  
- **XGBoost**
  - [XGBoost Model Directory](https://github.com/footcricket05/BCancerAI/tree/main/Models/Gradient%20Boosting)
  - XGBoost, an optimized gradient boosting algorithm, attains an accuracy of 95.61%. Its regularization techniques and parallel computing make it effective in accurate breast cancer prediction.

- **K-Means Clustering**
  - [K-Means Clustering Directory](https://github.com/footcricket05/BCancerAI/tree/main/Models/K-Means%20Clustering)
  - K-Means Clustering achieves an accuracy of 37.71% by grouping similar data points into clusters. While not well-suited for classification, K-Means is valuable for unsupervised learning tasks.
- **KNN**
  - [KNN Model Directory](https://github.com/footcricket05/BCancerAI/tree/main/Models/KNN)
  - KNN achieves an accuracy of 94.74% by classifying data points based on the majority class of their k-nearest neighbors. Its simplicity and intuitive concept make it a robust choice for breast cancer classification.

- **LightGBM**
  - [LightGBM Model Directory](https://github.com/footcricket05/BCancerAI/tree/main/Models/LightGBM)
  - LightGBM, a gradient boosting framework, achieves an accuracy of 96.49%. Its high efficiency, distributed computing support, and handling of large datasets contribute to its success in breast cancer classification.

- **Logistic Regression**
  - [Logistic Regression Model Directory](https://github.com/footcricket05/BCancerAI/tree/main/Models/Logistic%20Regression)
  - Logistic Regression attains an accuracy of 97.37% in breast cancer classification. Despite its simplicity, this linear model proves to be effective in capturing relationships between features.

- **Naive Bayes**
  - [Naive Bayes Model Directory](https://github.com/footcricket05/BCancerAI/tree/main/Models/Naive%20Bayes)
  - Naive Bayes achieves an accuracy of 96.49% by making probabilistic predictions based on the Bayes' theorem. Its simplicity and efficiency make it a valuable model for breast cancer classification.

- **Neural Network**
  - [Neural Network Model Directory](https://github.com/footcricket05/BCancerAI/tree/main/Models/Neural%20Network)
  - The Neural Network model, a sophisticated deep learning architecture, demonstrates an accuracy of 96.49%. Its ability to learn intricate patterns and hierarchical representations contributes to its success.

- **Random Forest**
  - [Random Forest Model Directory](https://github.com/footcricket05/BCancerAI/tree/main/Models/Random%20Forest)
  - The Random Forest model achieves an accuracy of 96.49% by leveraging an ensemble of decision trees. Its robustness, scalability, and ability to handle complex relationships make it a reliable choice.

- **SVM**
  - [SVM Model Directory](https://github.com/footcricket05/BCancerAI/tree/main/Models/SVM)
  - SVM achieves an accuracy of 95.61% by finding an optimal hyperplane to separate different classes. Its effectiveness in high-dimensional spaces contributes to accurate breast cancer classification.

- **Models with Principal Component Analysis (PCA)**
  - [Models with PCA Directory](https://github.com/footcricket05/BCancerAI/tree/main/Models/Models%20with%20PCA)
  - This directory contains models that have been enhanced with Principal Component Analysis (PCA), a dimensionality reduction technique. PCA is applied to the input features before training the models, allowing for more efficient training and potentially improved performance. The reduced dimensionality helps the models focus on the most relevant information, making them valuable tools for breast cancer classification. Explore the directory to access these PCA-enhanced models and leverage their capabilities for accurate and efficient breast cancer diagnosis.

Please note that these models are trained and evaluated on the [Diagnostic Wisconsin Breast Cancer Database](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data), a comprehensive collection of breast cancer images with corresponding labels.

## Usage 🚀

To use AI4BCancer's deep learning models, follow the instructions provided in each model's directory. Additionally, refer to the documentation for specific details on model training, evaluation, and integration into your workflow.

## Contributing 🤝

We welcome contributions from the community to enhance and improve AI4BCancer's deep learning models. If you have suggestions, want to report issues, or contribute enhancements, please follow the guidelines outlined in [CONTRIBUTING.md](CONTRIBUTING.md).

## License 📜

AI4BCancer is licensed under the [MIT License](LICENSE). By contributing, you agree to have your contributions covered by this license.
