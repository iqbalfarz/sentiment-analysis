import pickle
import re  # for regular expressions
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize  # for tokenizing the sentence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

nltk.download("punkt")

# helper function
def classification_report_custom(
    model, train_data, train_label, test_data, test_label, model_name="ML Model"
):
    """This function shows the classification report of given model on the given dataset."""

    ################ TRAIN ERROR and ACCURACY ###################
    # prediction of test dataset using best given Model
    test_predicted = model.predict(test_data)
    train_predicted = model.predict(train_data)

    train_accuracy = accuracy_score(train_label, train_predicted)
    test_accuracy = accuracy_score(test_label, test_predicted)

    train_f1_score = f1_score(train_label, train_predicted)
    test_f1_score = f1_score(test_label, test_predicted)

    print("TRAIN Accuracy : ", train_accuracy)
    print("TEST Accuracy : ", test_accuracy)
    print("=" * 50)
    print("TRAIN f1-score : ", train_f1_score)
    print("TEST f1-score : ", test_f1_score)
    ###########################################################################

    ############### Classification RESULT of both Train, and Test dataset ######
    train_cf_report = classification_report(train_label, train_predicted)
    test_cf_report = classification_report(test_label, test_predicted)

    print("-------------------------")
    print("| Classification Report |")
    print("-------------------------")
    print("TRAIN : ")
    print(train_cf_report)
    print("TEST : ")
    print(test_cf_report)
    print("-------------------------")

    #############################################################################################

    ################### ROC-AUC Score ###########################################################
    # getting train_score, and test_score
    test_prob = model.predict_proba(test_data)[:, 1]
    train_prob = model.predict_proba(train_data)[:, 1]

    # area under the curve
    train_auc = roc_auc_score(train_label, train_prob)
    test_auc = roc_auc_score(test_label, test_prob)

    ns_probs_train = [
        0 for _ in range(len(train_label))
    ]  # no skill probability for train

    ##########  TRAIN and TEST AUC  ###########
    fpr_train, tpr_train, _ = roc_curve(train_label, train_prob)
    fpr_test, tpr_test, _ = roc_curve(test_label, test_prob)

    ns_fpr, ns_tpr, _ = roc_curve(train_label, ns_probs_train)  # this is for no-skill

    plt.plot(fpr_train, tpr_train, label="TRAIN AUC Score={}".format(train_auc))
    plt.plot(fpr_test, tpr_test, label="TEST AUC Score={}".format(test_auc))
    plt.plot(ns_fpr, ns_tpr, label="No skill")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("TRAIN and TEST AUC Score of best {}".format(model_name))
    plt.legend()  # to show the label on plot
    plt.savefig("./results/LR_tfidf_auc.png")
    plt.show()  # force to show the plot


if __name__ == "__main__":
    ## 1. loading preprocessed dataset
    print(f"\n[INFO] 1. Loading dataset...\n")
    data_dir = Path("./dataset")

    # load train dataset
    train = pd.read_csv(data_dir / "train_new.csv")

    # load test dataset
    test = pd.read_csv(data_dir / "test_new.csv")

    print(f"sample of train dataset:\n{train.head()}")

    ## 2. Preprocessing the dataset
    # getting stopwords
    print(f"\n[INFO] 2. Preprocessing (lemmatizing and stopword removal) dataset...")

    STOP_WORDS = stopwords.words("english")
    STOP_WORDS.remove(
        "not"
    )  # removing "not" from STOPWORDS because it make sense in context of sentiment
    if "no" in STOP_WORDS:
        STOP_WORDS.remove("no")

    def preprocess(x):
        """
        this function preprocess the text including
        1. replace abbreviations like won't=> will not
        2. stemming removing=> remov
        3. stop-word removal
        4. removing html tags
        """
        # typecast to lowercase
        x = str(x).lower()

        # lemmatizer
        lemmatizer = WordNetLemmatizer()

        # remove digits, if any
        x = re.sub(r"[0-9]", "", x)

        # tokenize the sentence into words
        words = word_tokenize(x)  # It will return the list of words(tokens)

        # make final sentence including:
        # - lemmatization
        # - stopword removal
        words = [
            lemmatizer.lemmatize(word) for word in words if word not in STOP_WORDS
        ]  # list to store stemmed words

        return " ".join(words)

    # preprocess TRAIN, VAL, and TEST
    t0 = time()
    train["tweet"] = train["tweet"].apply(preprocess)
    print(
        f"[TIME] time taken to preprocess train data({train.shape[0]}): {time()-t0} s"
    )

    t0 = time()
    test["tweet"] = test["tweet"].apply(preprocess)
    print(f"[TIME] time taken to preprocess train data({test.shape[0]}): {time()-t0} s")

    # ## 3. split the dataset
    # X_train, X_val, y_train, y_val = train_test_split(train.tweet, train.sentiment_score, test_size=0.1, random_state=43, stratify=train.sentiment_score)
    # X_test, y_test = test.tweet, test.sentiment_score

    ## 3. Getting TF-IDF Vectorizer
    print(f"\n[INFO] 3. Getting TF-IDF Vectors...")

    def text_splitter(text):
        """
        this function split the text and return the list of splitted words
        """
        return text.split()

    t0 = time()
    tfidf_model_path = Path("./models/tfidf.pkl")
    retrain = True
    if tfidf_model_path.exists() and retrain is False:
        print(f"[INFO] Loading trained TF-IDF...")
        with open(tfidf_model_path, "rb") as tfidf_filepath:
            tfidf = pickle.load(tfidf_filepath)
    else:
        print(f"[INFO] Training TfidfVectorizer...")
        # initializing TF-IDF object
        tfidf = TfidfVectorizer(
            smooth_idf=True,
            tokenizer=text_splitter,
        )
        # fitting it on train dataset
        tfidf.fit(train["tweet"])
        with open(tfidf_model_path, "wb") as tfidf_filepath:
            pickle.dump(tfidf, tfidf_filepath)
        print(
            f"time taken to fit and save TfidfVectorizer on train dataset: {time()-t0} s"
        )

    # getting train and test tfidf vectors
    t0 = time()
    train_tfidf = tfidf.transform(train["tweet"])
    print(f"time taken to get TRAIN tf-idf feature vector: {time()-t0} s")

    t0 = time()
    test_tfidf = tfidf.transform(test["tweet"])
    print(f"time taken to get TEST tf-idf feature vector: {time()-t0} s")

    # getting train and test labels
    train_labels = train["sentiment_score"]
    test_labels = test["sentiment_score"]

    ## 4. Train Logistic Regression Model
    print(f"\n[INFO] 4. Training Logistic Regression...")
    lr_model_tfidf = LogisticRegression(max_iter=500)
    lr_model_tfidf.fit(train_tfidf, train_labels)
    print(f"time taken to train Logistic Regression on TF-IDF feautures: {time()-t0} s")

    # save the Logistic Regression model
    # saving the model for future use at given location
    lr_model_path = "./models/lr_model.pkl"
    with open(lr_model_path, "wb") as lr:
        pickle.dump(lr_model_tfidf, lr)

    ## 5. Getting Classification report
    print("\n[INFO] 5. Getting Classification report")
    classification_report_custom(
        model=lr_model_tfidf,
        train_data=train_tfidf,
        train_label=train_labels,
        test_data=test_tfidf,
        test_label=test_labels,
        model_name="Logistic Regression (TF-IDF)",
    )
    # print(f"Classification report: \n{classification_report_custom()}")
