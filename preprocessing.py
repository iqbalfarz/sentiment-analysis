"""
this module contains preprocessing raw data
"""
import re
import string
import warnings
from pathlib import Path
from time import time

import pandas as pd
import wordninja
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")


def cleanText(text):
    """
    process a single sentence
    """
    # remove html tags (if any)
    text = (BeautifulSoup(text)).get_text()

    # first of all replace abbreviations
    text = (
        text.replace("′", " ")
        .replace("’", " ")
        .replace(".", "")
        .replace("!", " ")
        .replace(",", " ")
        .replace("?", " ")
        .replace("won't", "will not")
        .replace("cannot", "can not")
        .replace("can't", "can not")
        .replace("n't", " not")
        .replace("what's", "what is")
        .replace("it's", "it is")
        .replace("'ve", " have")
        .replace("i'm", "i am")
        .replace("'re", " are")
        .replace("he's", "he is")
        .replace("that's", "that is")
        .replace("she's", "she is")
        .replace("'s", " own")
        .replace("'ll", " will")
        .replace("couldn't", "could not")
    )

    text = re.sub(r"@[A-Za-z0-9]+", "", text)  # removing @mentions
    text = re.sub(r"#", "", text)  # removing the "#" symbol
    text = re.sub(r"RT[\s]+", "", text)  # removing RT
    text = re.sub(r"https?:\/\/\S+", "", text)  # removing hyper links
    text = re.sub(r"\s+", " ", text)  # substituting multiple spaces into one

    # remove punctuations
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()

    # do a few other processing on words
    # by the observation we realized that there are a few words which
    # consists of very long length and the words like this
    # aaaaaaaaaaaahhhhhhhhhhhhhhh
    # wwwwaaaaaaaaaaaaaahhhhhhhhhhhhh
    # So, just process it
    sentence = []
    # Wordninja will split the text in a way like "iamagoodboy"-->["I","am","a","good","boy"]
    for word in text.split():
        # remove word of length greater than 17. Link :::: https://arxiv.org/ftp/arxiv/papers/1207/1207.2334.pdf
        if len(word) < 17:  # remove the words greater than len 17
            if len(word) > 6 and len(set(word)) <= 3:
                # then, do the processing like
                # wwwwaaaahhhh ---> wah
                temp_word = []
                prev_char = word[0]
                temp_word.append(prev_char)
                for character in list(word):
                    if character != prev_char:
                        temp_word.append(character)
                        prev_char = character

                sentence.append("".join(temp_word))
            elif len(word) > 2 and len(set(word)) <= 2:
                continue
            else:
                sentence.append(word)
    sentence = " ".join(sentence)
    sentence = " ".join(wordninja.split(sentence))
    return sentence


def preprocessing():
    """
    main function for preprocessing
    """
    time_start = time()
    ## 1. load the dataset
    dataset_dir = Path("./dataset")
    train_dataset_path = dataset_dir / "train.csv"
    test_dataset_path = dataset_dir / "test.csv"

    # load train dataset
    if train_dataset_path.exists():
        train = pd.read_csv(open(train_dataset_path, "r"), header=None)
    else:
        ValueError(f"{train_dataset_path} doesn't exist")

    # load test dataset
    if test_dataset_path.exists():
        test = pd.read_csv(open(test_dataset_path, "r"), header=None)
    else:
        ValueError(f"{test_dataset_path} doesn't exist")

    # update column names
    train = train[[0, 5]]
    test = test[[0, 5]]
    train.columns = ["sentiment_score", "tweet"]
    test.columns = ["sentiment_score", "tweet"]

    # Replace sentiment_score 4 by 1
    # replace values of columns by using DataFrame.loc[] property.
    train.loc[train["sentiment_score"] == 4, "sentiment_score"] = 1
    test.loc[test["sentiment_score"] == 4, "sentiment_score"] = 1

    ## remove 'score' `2` from test dataset
    test = test[test.sentiment_score.isin([0, 1])]

    ## apply preprocessing(cleanText) on train and test dataset
    t0 = time()
    train["tweet"] = train["tweet"].apply(cleanText)
    print(
        f"time taken to preprocess train({train.shape[0]} datapoints) dataset: {time()-t0} s"
    )

    t0 = time()
    test["tweet"] = test["tweet"].apply(cleanText)
    print(
        f"time taken to preprocess test({test.shape[0]} datapoints) dataset: {time()-t0} s"
    )

    # removing rows where there NaN in tweet column for both (train and test)
    train.dropna(inplace=True)
    test.dropna(inplace=True)

    # remove the rows where the value of tweet is empty string
    train.drop(train[train.tweet == ""].index, inplace=True)
    test.drop(test[test.tweet == ""].index, inplace=True)

    # save the model
    train.to_csv(open(dataset_dir / "train_new.csv", "wb"), index=None)
    test.to_csv(open(dataset_dir / "test_new.csv", "wb"), index=None)

    print(f"[RES] total time taken: {time()-time_start}")


if __name__ == "__main__":
    preprocessing()
