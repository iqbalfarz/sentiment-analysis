# Twitter Sentiment Analysis

## Problem Statement

Sentiment Analysis on twitter dataset as a binary classification using Machine Learning and Deep Learning.

## Data Source

- You can download the dataset using Kaggle [here!](https://www.kaggle.com/datasets/kazanova/sentiment140)
- Dataset contains 1.6M tweets
- Download the dataset, put it into `dataset/` directory, unzip and rename as `train.csv`.

## Machine Learning Modeling

- I trained classical Machine Learning Models like Logistic Regression on TF-IDF features, and LSA (Latent Semantic Analysis).
- I also trained two different Deep Learning models

  - CNN model using Conv1D with twitter pre-trained word vectors of `200d` (dimension).
  - LSTM-CNN model with same twitter pre-trained word vectors of `200d`.

- I also fine-tuned `DistilBERT` with a few additional `Dense` layer on top of the model.

## Metrics

- I used `accuracy`, `f1-score`, and `classification_report` to understand model performance.
- I also used ROC-AUC curve.

## Interpretation

- For Interpretation I used `LIME`(Local Interpretable Model-Agnostic Explanation) for both Machine Learning and Deep Learning models.

## Run & Installation

- Create a new virtual environment.
- Clone the repository.
- And install `requirements.txt` using below command.

```
pip install -r requirements.txt
```

- You will have to manually run each jupyter notebooks and see the result.

## Directory Structure

```
├───dataset
│   └───clustering.xlsx
├───jupyter_notebooks
│   ├───1.data_analysis_preprocessing.ipynb
│   ├───2.Sentiment Analysis using Machine Learning.ipynb
│   ├───3.twitter-sentiment-analysis-using-deep-learning.ipynb
│   ├───4.finetuning distilbert for sentiment classification.ipynb
├───models
│   ├─── README.md
├───result
│   ├─── containts resultant plots
│   ├─── README.md
├───README.md
├───.gitignore
├───finetune_distilbert.py
├───inference_cnn.py
├───inference_distilbert.py
├───inference_lr.py
├───inference_lstm_cnn.py
├───train_cnn.py
├───train_lr_on_tfidf.py
├───train_lstm_cnn.py
├───utils.py
├───requirements.txt
└───LICENSE
```
