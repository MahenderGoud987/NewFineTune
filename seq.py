import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model, load_model


# Set parameters
batch_size = 64
epochs = 100
latent_dim = 256
num_samples = 10000

# Read in data
data = pd.read_csv('chatbot_data.txt', sep='\t', header=None)
input_texts = data.iloc[:, 0].values.tolist()
target_texts = data.iloc[:, 1].values.tolist()

# Set max input and output lengths
max_input_length = max([len(text) for text in input_texts])
max_output_length = max([len(text) for text in target_texts])

# Tokenize input and output texts
tokenizer_inputs = Tokenizer()
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
word2idx_inputs = tokenizer_inputs.word_index
num_input_tokens = len(word2idx_inputs)

tokenizer_outputs = Tokenizer(filters='')
tokenizer_outputs.fit_on_texts(target_texts)
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
word2idx_outputs = tokenizer_outputs.word_index
num_output_tokens = len(word2idx_outputs)

# Define reverse target word index
reverse_target_word_index = dict((i, word) for word, i in word2idx_outputs.items())

# Pad input and output sequences to fixed length
encoder_inputs = pad_sequences(input_sequences, maxlen=max_input_length, padding='post')
decoder_inputs = pad_sequences(target_sequences, maxlen=max_output_length, padding='post')
decoder_targets = np.zeros((len(target_sequences), max_output_length, num_output_tokens), dtype='float32')
for i, target_sequence in enumerate(target_sequences):
    for j, token in enumerate(target_sequence):
        decoder_targets[i, j, token-1] = 1

# Define encoder and decoder models
encoder_inputs_placeholder = Input(shape=(max_input_length,))
x = keras.layers.Embedding(num_input_tokens+1, latent_dim)(encoder_inputs_placeholder)
encoder_outputs, h, c = LSTM(latent_dim, return_state=True)(x)
encoder_states = [h, c]

decoder_inputs_placeholder = Input(shape=(max_output_length,))
decoder_embedding = keras.layers.Embedding(num_output_tokens+1, latent_dim)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)
decoder_lstm = LSTM(latent_dim, return_state=True, return_sequences=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)
decoder_dense = Dense(num_output_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define training model
model = Model([encoder_inputs_placeholder, decoder_inputs_placeholder], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_inputs, decoder_inputs], decoder_targets, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# Save models and tokenizers
model.save('seq2seq_chatbot.h5')

encoder_model = Model(encoder_inputs_placeholder, encoder_states)
encoder_model.save('encoder_model.h5')

decoder_inputs_placeholder = Input(shape=(max_output_length,))
decoder_states_inputs = [Input(shape=(latent_dim,)), Input(shape=(latent_dim,))]
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs_x, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs_placeholder] + decoder_states_inputs, [decoder_outputs] + decoder_states)
decoder_model.save('decoder_model.h5')

with open('tokenizer_inputs.pickle', 'wb') as handle:
    pickle.dump(tokenizer_inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('tokenizer_outputs.pickle', 'wb') as handle:
    pickle.dump(tokenizer_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
