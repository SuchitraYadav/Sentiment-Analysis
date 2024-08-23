# IMDB Movie Reviews Sentiment Analysis

This project involves sentiment analysis of IMDB movie reviews using machine learning and deep learning techniques. The project includes data loading, exploratory data analysis, data cleaning, feature engineering, and model training using both traditional machine learning (Random Forest) and deep learning (LSTM) models.

## Project Overview

The goal of this project is to classify IMDB movie reviews into positive or negative sentiments based on the review text. 

The process includes:
Loading and Exploring the Dataset
Data Cleaning and Preprocessing
Feature Engineering
Model Training and Evaluation
Visualization

### 1. Loading and Exploring the Dataset
The dataset used is the IMDB movie reviews dataset, loaded using pandas. The code attempts to load the dataset from a CSV file. If the data is successfully loaded, it logs the success, otherwise, it logs the error.

### 2. Exploratory Data Analysis (EDA)
EDA is performed to understand the distribution of ratings, check for missing values, and visualize the class imbalance. 

This includes:
Checking for missing values.
Descriptive statistics of the ratings.
Visualizing the distribution of ratings.
Visualizing the most important words in positive and negative reviews using WordCloud.

### 3. Data Cleaning and Preprocessing
Data cleaning is a crucial step where the text is processed to remove noise and irrelevant information. 

The following operations are performed:
Remove special characters and URLs.
Stopwords removal: Custom stopwords are defined and applied.
Contraction expansion: Common contractions in English are expanded to their full form.
Tokenization and Lemmatization: Words are broken down into tokens, and each word is reduced to its base form (lemmatization).

### 4. Feature Engineering
In this step, the cleaned text data is transformed into numerical features using the TF-IDF vectorizer. The vectorizer converts the text into a matrix of TF-IDF features, which is then used as input for model training.

Binary Label Mapping: Ratings are converted into binary labels where reviews with ratings ≥ 7 are labeled as positive (1), and those with ratings ≤ 4 are labeled as negative (0).
TF-IDF Vectorization: The reviews are vectorized using TF-IDF, considering unigrams, bigrams, and trigrams.

### 5. Model Training and Evaluation
#### 5.1. Random Forest Classifier
A Random Forest Classifier is used for training on the TF-IDF features. The model is trained on the training set, and performance metrics such as precision, recall, F1-score, and ROC-AUC are evaluated on both the training and test sets.

Evaluation Metrics:

Precision: The ratio of correctly predicted positive observations to the total predicted positives.
AUC-ROC: Measures the area under the ROC curve.
F1 Score: Harmonic mean of precision and recall.

#### 5.2. LSTM Classifier
A deep learning model using Long Short-Term Memory (LSTM) is also implemented for sentiment classification. LSTM is chosen due to its effectiveness in handling sequential data, such as text.

The model architecture includes:

Embedding Layer: Converts words into dense vectors of fixed size.
LSTM Layers: Used to capture long-term dependencies in the sequence data.
Fully Connected Layers: For outputting the final classification results.

The model is trained using PyTorch, and metrics are computed similar to the Random Forest classifier.

## Logs and Debugging
Throughout the project, logging is used to keep track of the data processing steps, model training, and evaluation. This helps in debugging and understanding the flow of operations in the project.


## Requirements
Python 3.x
pandas
seaborn
matplotlib
plotly
scikit-learn
nltk
torch
wordcloud

## Conclusion
This project demonstrates a complete end-to-end pipeline for text classification using both machine learning and deep learning techniques. It includes essential steps such as data cleaning, feature engineering, and model evaluation, making it a robust solution for sentiment analysis.


