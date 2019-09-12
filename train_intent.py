import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import nltk
import re
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

def load_dataset(filename):
    df = pd.read_csv(filename, names = ["Sentence", "Intent"])
    print(df.head())
    intent = df["Intent"]
    unique_intent = list(set(intent))
    sentences = list(df["Sentence"])
    return (intent, unique_intent, sentences)

def cleaning(sentence):
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", sentence)
    w = word_tokenize(clean)
    clean_words = [i.lower() for i in w]
    return clean_words

def create_token(sentence):
    token = Tokenizer(filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~')
    token.fit_on_texts(sentence)
    return token

def preprocess(sentences, intent, unique_intent):
    # nltk.download("stopwords")
    # nltk.download("punkt")

    #define stemmer
    stemmer = LancasterStemmer()

    # input
    cleaned_words = []
    for s in sentences:
        words = cleaning(s)
        cleaned_words.append(words)
    with open("cleaned_words.txt", "wb") as fp:   #Pickling
        pickle.dump(cleaned_words, fp)
    word_tokenizer = create_token(cleaned_words)
    vocab_size = len(word_tokenizer.word_index) + 1
    max_length = len(max(cleaned_words, key = len))
    encoded_doc = word_tokenizer.texts_to_sequences(cleaned_words)
    padded_doc = pad_sequences(encoded_doc, maxlen = max_length, padding = "post")

    # output
    output_tokenizer = create_token(unique_intent)
    encoded_output = output_tokenizer.texts_to_sequences(intent)
    encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)
    o = OneHotEncoder(sparse = False)
    output_one_hot = o.fit_transform(encoded_output)

    input_data, label = padded_doc, output_one_hot
    print("Shape of X = %s and Y = %s" % (input_data.shape, label.shape))
    return input_data, label, max_length, vocab_size

intent, unique_intent, sentences = load_dataset("intent_data.csv")
X, Y, max_length, vocab_size  = preprocess(sentences, intent, unique_intent)

# ============ Define Model ===========
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length = max_length, trainable = False))
# model.add(Bidirectional(LSTM(128)))
model.add(LSTM(128))
model.add(Dense(32, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(Y.shape[1], activation = "softmax"))

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(X, Y, epochs = 500)
model.save('model.h5')


def predictions(text, max_length, classes):
    test_word = cleaning(text)
    with open("cleaned_words.txt", "rb") as fp:   # Unpickling
        cleaned_words = pickle.load(fp)
    test_tokenizer = create_token(cleaned_words)
    test_ls = test_tokenizer.texts_to_sequences(test_word)
    print(test_word)
    #Check for unknown words
    if [] in test_ls:
        test_ls = list(filter(None, test_ls)) 
    test_ls = np.array(test_ls).reshape(1, len(test_ls))
    x = pad_sequences(test_ls, maxlen = max_length, padding = "post")
    pred = model.predict_proba(x)
    predictions = pred[0]
    classes = np.array(classes)
    ids = np.argsort(-predictions)
    classes = classes[ids]
    predictions = -np.sort(-predictions)
    for i in range(pred.shape[1]):
        print("%s has confidence = %s" % (classes[i], (predictions[i])))

text = "what is the weather at muzium?"
predictions(text, max_length, unique_intent)
print('\n')

print(unique_intent)
print(max_length)

