"""
this module to get inference using CNN model
"""
import argparse
import pickle
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import wordninja
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import MAXLEN


def cleanText(text):

    # remove html tags (if any)
    text = (BeautifulSoup(text, features="lxml")).get_text()

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


def inference(query):
    """
    main method to get inference using CNN model
    """
    query = cleanText(query)

    # load tokenizer
    tokenizer_path = Path("./models/tokenizer.pkl")
    if tokenizer_path.exists():
        with open(tokenizer_path, "rb") as load_tokenizer:
            tokenizer = pickle.load(load_tokenizer)
    else:
        raise ValueError(f"[ERROR] {tokenizer_path} doesn't exist")

    # might be pd.Series(text)
    query_sequence = tokenizer.texts_to_sequences(query)

    # pad the sequence
    # maxlen should be equal to maxlen used in training CNN model
    # maxlen = 52 (used in script in the model)
    maxlen = MAXLEN

    padded_sequence = pad_sequences(query_sequence, padding="post", maxlen=maxlen)

    # load best CNN model
    model_path = Path("./models/best_cnn_model.hdf5")

    if model_path.exists():
        model = tf.keras.models.load_model(model_path)
    else:
        raise ValueError(f"[ERROR] {model_path} doesn't exist.")
    prediction = model.predict(padded_sequence)
    prob = np.argmax(prediction)

    return prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="query sentence/tweet",
        required=True,
    )
    args = parser.parse_args()

    print("Prediction: ", inference(args.input))
