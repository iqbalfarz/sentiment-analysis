import pickle
import argparse
import wordninja
from pathlib import Path
import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords # standard stopwords to remove from dataset
from nltk.stem import WordNetLemmatizer
import re

import nltk
nltk.download('punkt')

STOP_WORDS = stopwords.words('english')
STOP_WORDS.remove('not') # removing "not" from STOPWORDS because it make sense in context of sentiment

def text_splitter(text):
    """
    this function split the text and return the list of splitted words
    """
    return text.split()

def cleanText(text):
    # remove html tags (if any)
    text = BeautifulSoup(text, features='lxml')
    text = text.get_text()
    
    # first of all replace abbreviations
    text = text.replace("′", " ").replace("’", " ").replace(".","").replace("!"," ")\
                           .replace(",", " ").replace("?"," ")\
                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("that's","that is").replace("she's", "she is").replace("'s", " own")\
                           .replace("'ll", " will").replace("couldn't","could not")
    
    text = re.sub(r"@[A-Za-z0-9]+", "", text) # removing @mentions
    text = re.sub(r"#", "", text) # removing the "#" symbol
    text = re.sub(r"RT[\s]+", "", text) # removing RT
    text = re.sub(r"https?:\/\/\S+", "", text) # removing hyper links
    text = re.sub(r"\s+", " ", text) # substituting multiple spaces into one
    text = re.sub(r'[0-9]', '', text)
    
    # remove punctuations
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    
    # WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

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
        if len(word)<17: # remove the words greater than len 17
            if len(word)>6 and len(set(word))<=3:
                # then, do the processing like
                # wwwwaaaahhhh ---> wah
                temp_word = []
                prev_char = word[0]
                temp_word.append(prev_char)
                for character in list(word):
                    if character!=prev_char:
                        temp_word.append(character)
                        prev_char = character
                
                word = ''.join(temp_word)
                sentence.append(lemmatizer.lemmatize(word))
            elif len(word)>2 and len(set(word))<=2:
                continue
            else:
                sentence.append(lemmatizer.lemmatize(word))
    
    sentence = ' '.join(sentence)
    print("sentence: ", sentence)
    sentence = ' '.join(wordninja.split(sentence))
    return sentence

def inference(query):
    """
    this will return positive or negative probability
    """
    # preprocess the query
    query = cleanText(query)
    print(f"\nPreprocessed query: [{query}]")

    # load tfidfVectorizer
    tfidf_model_path = Path("./models/tfidf.pkl")
    with open(tfidf_model_path, "rb") as tfidf_filepath:
        tfidf = pickle.load(tfidf_filepath)

    tfidf.text_splitter = text_splitter

    query = (tfidf.transform([query]))[0]

    # load Logistic Regression model
    model_path = Path("./models/lr_model.pkl")
    with open(model_path, "rb") as load_model:
        model = pickle.load(load_model)

    prediction = model.predict(query)

    if prediction==0:
        return "Negative"
    return "Positive"

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        help="query sentence/tweet",
        required=True,
    )
    args = parser.parse_args()

    print("Prediction: ",inference(args.input))
