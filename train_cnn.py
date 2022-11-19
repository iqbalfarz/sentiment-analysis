"""
this module is to train CNN (Conv1D) model for twitter Sentiment Analysis
"""
# Imports
import warnings

warnings.filterwarnings("ignore")  # ignore warnings

import pickle  # to save model
from pathlib import Path
from time import time

import matplotlib.pyplot as plt  # visualization
import numpy as np  # numerical python processing(linear algebra)
import pandas as pd  # data preprocessing
import seaborn as sns  # Visualization on top of Matplotlib
import tensorflow as tf

# metrics to assess the model performance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from utils import MAXLEN


### Helper functions ###
def classification_report_dl(model, test_data, test_label, model_name="DL Model"):
    """This function shows the classification report of given model on the given dataset."""

    ################ TEST ERROR and ACCURACY ###################
    # prediction of test dataset using best given Model
    test_predicted = pd.Series((model.predict(test_data)).flatten()).apply(
        lambda x: 1 if x > 0.5 else 0
    )

    test_accuracy = accuracy_score(test_label, test_predicted)

    test_f1_score = f1_score(test_label, test_predicted)

    print("TEST Accuracy : ", test_accuracy)
    print("=" * 50)
    print("TEST f1-score : ", test_f1_score)
    ###########################################################################

    ############### CLASSIFICATIO REPORT for TEST dataset ######
    test_cf_report = classification_report(test_label, test_predicted)

    print("-------------------------")
    print("| Classification Report |")
    print("-------------------------")
    print("TEST : ")
    print(test_cf_report)
    print("-------------------------")

    #############################################################################################

    ################### ROC-AUC Score ###########################################################
    # getting train_score, and test_score
    test_prob = model.predict(test_data)

    # area under the curve
    test_auc = roc_auc_score(test_label, test_prob)

    ns_probs_test = [
        0 for _ in range(len(test_label))
    ]  # no skill probability for test dataset

    ##########  TRAIN AUC  ###########
    fpr_test, tpr_test, _ = roc_curve(test_label, test_prob)

    ns_fpr, ns_tpr, _ = roc_curve(test_label, ns_probs_test)  # this is for no-skill

    plt.plot(fpr_test, tpr_test, label="TEST AUC Score={}".format(test_auc))
    plt.plot(ns_fpr, ns_tpr, label="No skill")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("TEST AUC Score of best {}".format(model_name))
    plt.legend()  # to show the label on plot
    plt.savefig(f"./results/cnn.png")
    plt.show()  # force to show the plot


if __name__ == "__main__":
    root_dir = Path(".")
    dataset_dir = root_dir / "dataset"

    ## 1. load dataset
    print("\n[INFO] 1. Loading dataset...")
    train = pd.read_csv(dataset_dir / "train_new.csv")
    test = pd.read_csv(dataset_dir / "test_new.csv")

    print(f"\nSample of train data: \n{train.head()}")

    ## 2. Splitting dataset into TRAIN, VAD, and TEST
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        train.tweet,
        train.sentiment_score,
        test_size=0.1,
        random_state=42,
        stratify=train.sentiment_score,
    )
    X_test, y_test = test.tweet, test.sentiment_score

    # no.of datapoints in train and test dataset
    print("\nNo. of datapoint in TRAIN : ", X_train.shape[0])
    print("No. of datapoint in VAL  : ", X_val.shape[0])
    print("No. of datapoint in TEST  : ", X_test.shape[0])

    ## 3. tokenization
    # which can be used by embeddings
    # tokenize the data that can be used by embeddings
    print(f"\n[INFO] tokenizing dataset...")

    tokenizer_path = root_dir / "models/tokenizer.pkl"
    retrain = False
    if tokenizer_path.exists() and retrain is False:
        print("[INFO] Loading saved tokenizer")
        with open(tokenizer_path, "rb") as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)
    else:
        t0 = time()
        print("[INFO] fitting tokenizer on TRAIN data...")
        tokenizer = Tokenizer(lower=True)  # used in the research paper
        tokenizer.fit_on_texts(X_train.apply(lambda x: np.str_(x)))
        # save the tokenizer for future use
        with open(tokenizer_path, "wb") as tokenizer_file:
            pickle.dump(tokenizer, tokenizer_file, protocol=pickle.HIGHEST_PROTOCOL)
            # pickle.HIGHEST_PROTOCOL is highest protocol version available
        print(f"[RES] time taken to fit and save the tokenizer: {time()-t0} s")

    print(f"\nNo.of words in tokenizer: {len(tokenizer.word_index)}\n")

    # changing texts into sequences to train word embeddings
    print(f"\n[INFO] Changing texts to sequences...")
    t0 = time()
    X_train = tokenizer.texts_to_sequences(X_train)
    X_val = tokenizer.texts_to_sequences(X_val)
    X_test = tokenizer.texts_to_sequences(X_test)
    print(
        f"\n[RES] time taken to get sequences for TRAIN, VAL, and TEST: {time() - t0} s"
    )

    # get vocabulary size
    vocab_size = len(tokenizer.word_index) + 1  # adding 1 because 0 is reserved
    print(f"\nVocabulary size: {vocab_size}\n")

    ## 4. Padding
    # problem is: each sentence has variable no.of words.
    # So, do padding.
    maxlen = MAXLEN  # after processing we observe the maxlen is 52
    print(f"\n[INFO] Padding to maxlen {maxlen}...")
    # It doesn't matter that you preprend or append the padding
    t0 = time()
    X_train = pad_sequences(X_train, padding="post", maxlen=maxlen)
    X_val = pad_sequences(X_val, padding="post", maxlen=maxlen)
    X_test = pad_sequences(X_test, padding="post", maxlen=maxlen)
    print(
        f"\n[RES] time taken to get the padding for TRAIN, VAL, and TEST: {time()-t0} s"
    )

    ## 5. Create embedding matrix
    embedding_dim = 200
    print(
        f"\n[INFO] Creating Embedding Matrix for embedding dimension: {embedding_dim} ..."
    )

    def create_embedding_matrix(filepath, word_index, embed_dim):
        """
        This function creates the embedding matrix.
        """
        vocab_size = len(word_index) + 1  # Adding 1 because of reserved 0 index
        embedding_matrix = np.zeros((vocab_size, embedding_dim))

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:  # for each row in the file
                word, *vector = line.split()
                if word in word_index:
                    idx = word_index[word]
                    embedding_matrix[idx] = np.array(vector, dtype=np.float32)[
                        :embed_dim
                    ]

        return embedding_matrix

    t0 = time()
    filepath = root_dir / "dataset" / "glove.twitter.27B.200d.txt"
    embedding_matrix = create_embedding_matrix(
        filepath, tokenizer.word_index, embedding_dim
    )
    print(
        f"\n[RES] time taken to create embedding_matrix{embedding_dim}: {time()-t0} s"
    )

    ## 6. Callbacks
    # Path to save the LSTM-CNN model
    checkpoint_path_cnn = root_dir / "models/best_cnn_model.hdf5"

    # Callbacks to save model
    save_cnn_model = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path_cnn,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
        mode="max",  # save model when get max f1-score
    )

    # Early stop
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        mode="auto",
    )

    ## 7. Creating Model
    print(f"\n[INFO] Creating CNN Model...")
    # https://keras.io/layers/embeddings/
    # Now, we will learn a new embedding space using Embedding layer
    # which maps above encoded word representation into dense vector.
    # input_dim = the size of the vocabulary
    # output_dim = the size of the dense vector
    # input_length = the length of the sequence

    # clear all previous sessions
    tf.keras.backend.clear_session()
    embedding_dim = 200

    def get_compiled_cnn_model():
        model = Sequential()
        model.add(
            layers.Embedding(
                input_dim=vocab_size,  # vocab_size (no.of words)
                output_dim=embedding_dim,  # embedding dimension
                input_length=maxlen,
                trainable=False,
                weights=[embedding_matrix],
            )
        )
        model.add(
            layers.Conv1D(512, 5, activation="relu", kernel_initializer="he_normal")
        )
        model.add(
            layers.Conv1D(256, 5, activation="relu", kernel_initializer="he_normal")
        )
        model.add(
            layers.Conv1D(128, 5, activation="relu", kernel_initializer="he_normal")
        )
        model.add(
            layers.Conv1D(128, 5, activation="relu", kernel_initializer="he_normal")
        )
        model.add(
            layers.Conv1D(64, 5, activation="relu", kernel_initializer="he_normal")
        )
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dense(100, activation="relu"))
        model.add(layers.Dense(10, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))

        # compile the model
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    # # create a mirrored Strategy
    # strategy = tf.distribute.MirroredStrategy()
    # # Open a strategy scope.
    # with strategy.scope():
    #     # Everything that creates variables should be under the strategy scope.
    #     # In general this is only model construction & `compile()`.
    #     lstm_model = get_compiled_lstm_model()
    cnn_model = get_compiled_cnn_model()
    print(f"\n[INFO] Summary of the model.")
    cnn_model.summary()

    print(f"\n[INFO] Training CNN model...")
    cnn_model.fit(
        X_train,
        y_train,
        epochs=20,
        validation_data=(X_val, y_val),
        batch_size=128,
        verbose=1,
        callbacks=[save_cnn_model, early_stop],
    )
    print(f"time taken to train CNN model: {time()-t0} s")
