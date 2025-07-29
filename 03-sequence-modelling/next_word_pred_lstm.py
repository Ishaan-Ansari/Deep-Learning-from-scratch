import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import tokenizer

with open('./text.txt') as f:
    text = f.read()

# for sentences in text.split('\n'):
#     print(sentences)

# convert words to vecs
input_squences = []
for sentence in text.split('\n'):
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]

    for i in range(1, len(tokenized_sentence)):
        input_squences.append(tokenized_sentence[:i+1])

max_len = max([len(x) for x in input_squences])

from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_input_sequences = pad_sequences(input_squences, maxlen=max_len, padding='pre')

X = padded_input_sequences[:, :-1]
y = padded_input_sequences[:, -1]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes=283)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()
model.add(Embedding(283, 100, input_length=56))
model.add(LSTM(150))
model.add(Dense(150))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(X, y, epochs=100, batch_size=10)

