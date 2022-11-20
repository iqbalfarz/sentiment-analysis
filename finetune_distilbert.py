"""
In this module we fine-tune DistilBERT for twitter sentiment Analysis
"""

from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast

# Define the maximum number of words to tokenizer (DistilBERT can tokenizer upto 512)
# because maximum tweet length can be of 280 words only.
MAX_LENGTH = 60  # because of memory issue, we are using this.
# and the maxlen in the dataset is 52 (after preprocessing)

EPOCHS = 1
BATCH_SIZE = 128  # memory constraints
LEARNING_RATE = 5e-5  # recommended by BERT author's
RANDOM_STATE = 42

root_dir = Path("./")

# Define function to encode text data in batches
def batch_encode(tokenizer, texts, batch_size=128, max_length=MAX_LENGTH):
    """
    A function to encode batch of text and returns the texts'corresponding encodings
    and attention masks that are ready to be fed into a pre-trained transformer model.

    Parameters
    ----------
        tokenizer: ``PreTrainedTokenizer``
            Tokenizer object from the PreTrainedTokenizer class
        texts: ``List[str]``
            List of strings where each string represents a text
        batch_size: ``int``
            size of each batch
        max_length: ``int``
            maximum length of the sentence.

    Returns
    -------
        input_ids: ``tf.Tensor``
            sequence of texts encoded as `tf.Tensor` object
        attention_mask:
            the texts' attention mask encoded as a `tf.Tensor` object.
    """
    input_ids = []
    attention_mask = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=batch,
            max_length=max_length,
            padding="max_length",
            # truncate to a maximum length specified with
            # the argument max_length or to the maximum acceptable
            # input length for thee model if that argument is not provided.
            truncation="only_first",
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        input_ids.extend(inputs["input_ids"])
        attention_mask.extend(inputs["attention_mask"])

    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)


def build_model(transformer, max_length=MAX_LENGTH):
    """
    Template for building a model on top of BERT or DistilBERT architecture
    for a binary classification task.

    Parameters
    ----------
        transformer:
            A base Huggingface transformer model object (BERT or DistilBERT)
            with no added classification head on top of it.
        max_length: int
            maximum number of encoded tokens in a given sequence.

    Returns
    -------
        model:
            A compiled `tf.keras.Model` with added classification layers
            on top of the base pre-trained model architecture.
    """

    # define weight initializer with a random sed to ensure reproducibility
    weight_initializer = tf.keras.initializers.GlorotNormal(seed=RANDOM_STATE)

    # define input layers
    input_ids_layer = tf.keras.layers.Input(
        shape=(max_length,),
        name="input_ids",
        dtype="int32",
    )
    input_attention_layer = tf.keras.layers.Input(
        shape=(max_length,), name="input_attention", dtype="int16"
    )

    # DistilBERT outputs a tuple where the first element at index 0
    # represents the hidden-state at the output of the model's last layer.
    # It is a tf.Tensor of shape (batch_size, sequence_length, hidden_size=768).
    last_hidden_state = transformer([input_ids_layer, input_attention_layer])[0]

    # We only care about DistilBERT's output for the [CLS] token,
    # which is located at index 0 of every encoded sequence.
    # Slicing out the [CLS] tokens gives us 2D data.
    # [CLS] stands for Classification
    cls_token = last_hidden_state[:, 0, :]

    #                                                 ##
    # Define additional dropout and dense layers here ##
    #                                                 ##

    # Define a single node that makes up the output layer (for binary classification)
    dense1 = tf.keras.layers.Dense(
        200,
        activation="relu",
        kernel_initializer="he_normal",
    )(cls_token)
    dense2 = tf.keras.layers.Dense(
        100,
        activation="relu",
        kernel_initializer="he_normal",
    )(dense1)
    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer=weight_initializer,
        kernel_constraint=None,
        bias_initializer="zeros",
    )(dense2)

    # Define the model
    model = tf.keras.Model([input_ids_layer, input_attention_layer], output)

    # Compile the model
    model.compile(
        tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def classification_report_dl(
    model, test_data, test_label, model_name="DistilBERT Model"
):
    """This function shows the classification report of given model on the given dataset."""
    ################ TEST  ACCURACY ###################
    # prediction of test dataset using best given Model
    test_predicted = pd.Series((model.predict(test_data)).flatten()).apply(
        lambda x: 1 if x > 0.5 else 0
    )
    test_accuracy = accuracy_score(test_label, test_predicted)
    test_f1_score = f1_score(test_label, test_predicted)
    print("TEST Accuracy : ", test_accuracy)
    print("=" * 50)
    print("TEST f1-score : ", test_f1_score)

    ############### CLASSIFICATIO RESULT for Test dataset ######
    test_cf_report = classification_report(test_label, test_predicted)

    print("-------------------------")
    print("| Classification Report |")
    print("-------------------------")
    print("TEST : ")
    print(test_cf_report)
    print("-------------------------")
    ################### ROC-AUC Score ###########################################################
    # getting test_score
    test_prob = model.predict(test_data)
    # area under the curve
    test_auc = roc_auc_score(test_label, test_prob)
    ns_probs_test = [
        0 for _ in range(len(test_label))
    ]  # no skill probability for test dataset
    ##########  TEST AUC  ###########
    fpr_test, tpr_test, _ = roc_curve(test_label, test_prob)
    ns_fpr, ns_tpr, _ = roc_curve(test_label, ns_probs_test)  # this is for no-skill

    plt.plot(fpr_test, tpr_test, label="TEST AUC Score={}".format(test_auc))
    plt.plot(ns_fpr, ns_tpr, label="No skill")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("TEST AUC Score of best {}".format(model_name))
    plt.legend()  # to show the label on plot
    results_dir = root_dir / "results"
    if not results_dir.exists():
        results_dir.mkdir()
    plt.savefig("./results/distil.png")
    plt.show()  # force to show the plot


def finetune_distilbert():
    """
    this is the main function to finetune DistilBERT
    """
    ## 1. Loading dataset

    print(f"\n[INFO] 1. Loading dataset...")
    dataset_dir = root_dir / "dataset"
    train = pd.read_csv(dataset_dir / "train_new.csv")
    test = pd.read_csv(dataset_dir / "test_new.csv")

    print(f"\nSample of TRAIN datapoints:\n{train.head()}")

    ## 2. Splitting the data into TRAIN, VAL, and TEST
    print(f"[INFO] 2. Splitting the dataset...")
    t0 = time()
    X_train, X_val, y_train, y_val = train_test_split(
        train.tweet,
        train.sentiment_score,
        # train_size=0.6,
        test_size=0.02,
        random_state=43,
        stratify=train.sentiment_score,
    )
    X_test, y_test = test.tweet, test.sentiment_score
    print(f"[RES] time taken to split into TRAIN and VAL: {time()-t0} s")

    # no.of datapoints in train, validation and test dataset
    print("\nNo. of datapoint in TRAIN : ", X_train.shape[0])
    print("No. of datapoint in VAL  : ", X_val.shape[0])
    print("No. of datapoint in TEST  : ", X_test.shape[0])

    ## 3. Tokenization
    # tokenization is a process to make sentence in the form
    # which is expected(can be processed) by Model.
    print(f"\n[INFO] 3. Tokenizing dataset")
    checkpoint = "distilbert-base-uncased"
    # Instantiate DistilBert tokenizer. We use the faster version to optimizer runtime
    tokenizer = DistilBertTokenizerFast.from_pretrained(checkpoint)

    t0 = time()
    # Encode X_train
    X_train_ids, X_train_attention = batch_encode(tokenizer, X_train.tolist())
    # Encode X_val
    X_val_ids, X_val_attention = batch_encode(tokenizer, X_val.tolist())
    # Encode X_test
    X_test_ids, X_test_attention = batch_encode(tokenizer, X_test.tolist())
    print(
        f"[RES] time taken to get inputs ids and attention masks for TRAIN, VAL, and TEST: {time()-t0} s"
    )

    ## 4. Defining a Model Architecture
    print(f"[INFO] 4. Defining Model Architecture...")
    from transformers import DistilBertConfig, TFDistilBertModel

    DISTILBERT_DROPOUT = 0.2  # default is 0.1
    DISTILBERT_ATT_DROPOUT = 0.2  # default is 0.1

    # Configure DistilBERT's initialization
    config = DistilBertConfig(
        dropout=DISTILBERT_DROPOUT,
        attention_dropout=DISTILBERT_ATT_DROPOUT,
        output_hidden_states=True,
    )

    # pre-trained DistilBERT transformer model will output raw hidden-states
    # and without any specific head on top. So, we are suppose to use this for our downstream task.
    # DistilBERT model will be initialized by our custom config (configuration)
    distilBERT = TFDistilBertModel.from_pretrained(checkpoint, config=config)

    # freeze the DistilBERT layers (means untrainable)
    # we can later unfreeze when model performance converges.
    for layer in distilBERT.layers:
        layer.trainable = False

    ### Add a Classification head on top of DistilBERT Classifier.
    # Path to save the distilbert model
    model_dir = root_dir / "models"
    checkpoint_path_distilbert = model_dir / "best_distil_model.tf"

    if not model_dir.exists():
        model_dir.mkdir()

    # Callbacks to save model
    save_distilbert_model = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path_distilbert,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
        mode="max",  # save model when get max validation accuracy
    )

    # Early stop
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        mode="auto",
    )

    # clear previous session
    tf.keras.backend.clear_session()
    model = build_model(
        distilBERT,
    )
    print(f"\n[INFO] Model Summary: {model.summary()}")

    ## 5. Training Classification layers
    print(f"\n[INFO] Training Classification Layer Weights")
    NUM_STEPS = len(X_train.index) // BATCH_SIZE
    # Train the model
    model.fit(
        x=[X_train_ids, X_train_attention],
        y=y_train.to_numpy(),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        steps_per_epoch=NUM_STEPS,
        validation_data=([X_val_ids, X_val_attention], y_val.to_numpy()),
        verbose=1,
        callbacks=[save_distilbert_model, early_stop],
    )

    ## 6. Get the classification result
    best_distil_model = tf.keras.models.load_model("./models/best_distil_model.tf")

    classification_report_dl(
        model=best_distil_model,
        test_data=[X_test_ids, X_test_attention],
        test_label=y_test.to_numpy(),
        model_name="DistilBERT model",
    )


if __name__ == "__main__":
    finetune_distilbert()
