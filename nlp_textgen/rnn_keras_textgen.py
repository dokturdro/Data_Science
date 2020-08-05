from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import pandas as pd
import numpy as np
import string, os

# load the data file
file = open(r"C:\Users\Administrator\Desktop\datasets\text\fsfitzgerald.txt", 'r')
text = file.read()
file.close()

# tokenize words while excluding non-alphabetic characters
words = word_tokenize(text)
words = [word.lower() for word in words if word.isalpha()]

# exclude common stopwords that lower prediction accuracy
stopwords = set(stopwords.words('english'))
words = [word for word in words if word not in stopwords]

# exclude bottom 1000 rarest words
freq = pd.Series.value_counts(words)[-1000:]
freq = set(freq.index)
words = [word for word in words if word not in freq]

# lemmatize the words for its cores
lemma = WordNetLemmatizer()
words = [lemma.lemmatize(word) for word in words]
tokenizer = Tokenizer(lower=True)
tokenizer.fit_on_texts([words])
encoded = tokenizer.texts_to_sequences([words])[0]

# setup global size of the network, ngrams length, check the vocabsize
rnn_size = 256
epochs = 32
n_gram = 3
vocab_size = len(tokenizer.word_index) + 1

# populate the list with sequences of word tokens, check its size
sequences = list()
for i in range(1, vocab_size):
	sequence = encoded[i-1:i+(n_gram-1)]
	sequences.append(sequence)
# setup an array to split for training
sequences = np.array(sequences)
X, y = sequences[:,0],sequences[:,1]

# one hot encoding
y = to_categorical(y, num_classes=vocab_size)

# define keras model with Embedding representing one hot matrix
model = Sequential()
model.add(Embedding(vocab_size, 256, input_length=1))
model.add(Dropout(0.4))
model.add(LSTM(rnn_size))
model.add(Dropout(0.2))
model.add(Dense(vocab_size, activation='softmax'))

# compile and fit the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=epochs, verbose=2)

# setup a function to be called for generation of text
def generate_seq(model, tokenizer, seed_text, n_words):
	in_text, result = seed_text, seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		print(encoded)
		encoded = np.array(encoded)
		print(encoded)
		# predict a word in the vocabulary
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text, result = out_word, result + ' ' + out_word
	return result

# generate a sequence beginnign with a desired word and word length
print(generate_seq(model, tokenizer, 'stand', 6))