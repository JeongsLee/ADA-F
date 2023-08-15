import keras_nlp
import pathlib
import random
random.seed(1234)

import tensorflow as tf
tf.random.set_seed(1234)
import numpy as np
np.random.seed(1234)

from tensorflow import keras
keras.utils.set_random_seed(1234)
from tensorflow_text.tools.wordpiece_vocab import (
    bert_vocab_from_dataset as bert_vocab,
)
import os

import numpy as np
np.random.seed(1234)

import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
tf.random.set_seed(1234)
tf.get_logger().setLevel('ERROR')

from tensorflow import keras
#import tensorflow_hub as hub
import tensorflow_datasets as tfds

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers



class ADAF(tf.keras.layers.Layer):
    def __init__(self,  N_p = 5, N_m = 5, L=1.,  DTYPE='float32', kernel_regularizer=None):
        super(ADAF, self).__init__()        
        self.N_p = N_p
        self.N_m = N_m
        self.L = L
        self.x_i = tf.cast(tf.linspace(0., L, N_p+1),dtype=DTYPE)
        self.DTYPE = DTYPE
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    def build(self, input_shape):        

        self.init1 = self.add_weight('init1', shape=(), initializer='random_normal', regularizer = self.kernel_regularizer, trainable=True, dtype=self.DTYPE)
        self.init2 = self.add_weight('init2', shape=(), initializer='random_normal', regularizer = self.kernel_regularizer, trainable=True, dtype=self.DTYPE)        
        self.init3 = self.add_weight('init3', shape=(), initializer='random_normal', regularizer = self.kernel_regularizer, trainable=True, dtype=self.DTYPE)
        self.init4 = self.add_weight('init4', shape=(), initializer='random_normal', regularizer = self.kernel_regularizer, trainable=True, dtype=self.DTYPE)                           
            


        self.alpha = self.add_weight('alpha', shape=(), initializer='random_normal', regularizer = self.kernel_regularizer, trainable=True, dtype=self.DTYPE)

        self.beta = self.add_weight('beta', shape=(), initializer='random_normal', regularizer = self.kernel_regularizer, trainable=True, dtype=self.DTYPE)
        
        self.W_i = self.add_weight('W_i', shape=(self.N_p,), initializer='random_normal', regularizer = self.kernel_regularizer, trainable=True, dtype=self.DTYPE)               

    def out_an(self, n, x_1, x_2, W_i):
        if n == 0:
            a_n = tf.reduce_sum(W_i)
            a_n = a_n/self.N_p
        else:
            sum_1 = tf.math.sin(n*np.pi/self.L* x_1)
            sum_2 = -tf.math.sin(n*np.pi/self.L* x_2)
            a_n = W_i * (sum_1 + sum_2)
            a_n = tf.reduce_sum(a_n)
            a_n = (2./(n*np.pi)) * a_n
        return a_n 
    def out_bn(self, n, x_1, x_2, W_i):                
        sum_1 = -tf.math.cos(n*np.pi/self.L* x_1)
        sum_2 = tf.math.cos(n*np.pi/self.L* x_2)        
        b_n = W_i *(sum_1 + sum_2)
        b_n = tf.reduce_sum(b_n)
        b_n = (2./ (n*np.pi))*b_n
        return b_n
    def out_g_x_0(self, x):           
        x_1 = self.x_i[1:]
        x_2 = self.x_i[:-1]      
               
        g_x = tf.cast(0., self.DTYPE)
        g_x += (self.alpha)*self.out_an(0, x_1, x_2, self.W_i)*x
        for n in range(1,self.N_m+1):
            factor = self.L/(n*np.pi)
            factor = tf.constant(factor,self.DTYPE)            
            g_x += (1.-self.alpha)*(-factor*self.out_bn(n, x_1, x_2, self.W_i)) * ( tf.math.cos(x/factor) - 1 )                        
            g_x += (self.alpha)*(factor*self.out_an(n,x_1, x_2, self.W_i)) * (tf.math.sin(x/factor))
        g_x += self.init1
        return g_x
    def out_g_x_1(self, x):           
        x_1 = self.x_i[1:]
        x_2 = self.x_i[:-1]                
                
        g_x = tf.cast(0., self.DTYPE)
        g_x += (self.alpha)*self.out_an(0, x_1, x_2, self.W_i)/2. * tf.math.square(x)
        for n in range(1,self.N_m+1):
            factor = self.L/(n*np.pi)
            factor = tf.constant(factor,self.DTYPE)            
            g_x += (1.-self.alpha)*(-factor*self.out_bn(n, x_1, x_2, self.W_i)) * ( (factor)*tf.math.sin(x/factor) - x ) 
            g_x += (self.alpha)*tf.math.square(factor)*self.out_an(n, x_1, x_2, self.W_i)*(1.-tf.math.cos(x/factor))
        g_x += self.init1*x + self.init2
        return g_x
    def out_g_x_2(self, x):
        if x.dtype == self.dtype:
            pass
        else:        
            x = tf.cast(x,self.dtype)        
        x_1 = self.x_i[1:]
        x_2 = self.x_i[:-1]
        g_x = tf.cast(0., self.dtype)
        g_x += (self.alpha)*self.out_an(0, x_1, x_2, self.W_i)/6. * tf.math.pow(x,3)
        for n in range(1,self.N_m+1):
            factor = self.L/(n*np.pi)
            factor = tf.constant(factor,self.dtype)            
            g_x += (1.-self.alpha)*(factor*self.out_bn(n, x_1, x_2, self.W_i))*(0.5*tf.math.square(x) + tf.math.pow(factor,2)*(tf.math.cos(x/factor)-1.))
            g_x += (self.alpha)*tf.math.square(factor)*self.out_an(n, x_1, x_2, self.W_i)*(x-factor*tf.math.sin(x/factor))
        g_x += 0.5*self.init1*tf.math.square(x) + self.init2*x + self.init3
        return g_x        
    def out_g_x_3(self, x):        
        x_1 = self.x_i[1:]
        x_2 = self.x_i[:-1]
        g_x = tf.cast(0., self.DTYPE)
        g_x += (self.alpha)*self.out_an(0, x_1, x_2, self.W_i)/24. * tf.math.pow(x,4)
        for n in range(1,self.N_m+1):
            factor = self.L/(n*np.pi)
            factor = tf.constant(factor,self.DTYPE)            
            g_x += (1.-self.alpha)*(factor*self.out_bn(n, x_1, x_2, self.W_i))*(tf.math.pow(x,3)/6.-tf.math.square(factor)*x+tf.math.pow(factor,3)*tf.math.sin(x/factor))
            g_x += (self.alpha)*tf.math.square(factor)*self.out_an(n, x_1, x_2, self.W_i)*(0.5*tf.math.square(x)+tf.math.square(factor)*(tf.math.cos(x/factor)-1.))
        g_x += self.init1*tf.math.pow(x,3)/6. + self.init2*tf.math.square(x)/2. + self.init3*x + self.init4        
        return g_x
    def call(self, inputs):
        if inputs.dtype == self.DTYPE:
            pass
        else:        
            inputs = tf.cast(inputs,self.DTYPE)
        inputs = inputs + self.L/2.
        return self.out_g_x_3(inputs) + tf.math.multiply(self.beta*self.out_g_x_2(inputs),inputs)

BATCH_SIZE = 64
EPOCHS = 40  # This should be at least 10 for convergence
MAX_SEQUENCE_LENGTH = 40
ENG_VOCAB_SIZE = 15000
SPA_VOCAB_SIZE = 15000

EMBED_DIM = 256
INTERMEDIATE_DIM = 2048
NUM_HEADS = 8

text_file = keras.utils.get_file(
    fname="spa-eng.zip",
    origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
    extract=True,
)
text_file = pathlib.Path(text_file).parent / "spa-eng" / "spa.txt"

with open(text_file, encoding='UTF8') as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    eng, spa = line.split("\t")
    eng = eng.lower()
    spa = spa.lower()
    text_pairs.append((eng, spa))
    

random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")    

def train_word_piece(text_samples, vocab_size, reserved_tokens):
    word_piece_ds = tf.data.Dataset.from_tensor_slices(text_samples)
    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        word_piece_ds.batch(1000).prefetch(2),
        vocabulary_size=vocab_size,
        reserved_tokens=reserved_tokens,
    )
    return vocab
    
    
reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

eng_samples = [text_pair[0] for text_pair in train_pairs]
eng_vocab = train_word_piece(eng_samples, ENG_VOCAB_SIZE, reserved_tokens)

spa_samples = [text_pair[1] for text_pair in train_pairs]
spa_vocab = train_word_piece(spa_samples, SPA_VOCAB_SIZE, reserved_tokens)   

print("English Tokens: ", eng_vocab[100:110])
print("Spanish Tokens: ", spa_vocab[100:110])

eng_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=eng_vocab, lowercase=False
)
spa_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=spa_vocab, lowercase=False
)


def preprocess_batch(eng, spa):
    batch_size = tf.shape(spa)[0]

    eng = eng_tokenizer(eng)
    spa = spa_tokenizer(spa)

    # Pad `eng` to `MAX_SEQUENCE_LENGTH`.
    eng_start_end_packer = keras_nlp.layers.StartEndPacker(
        sequence_length=MAX_SEQUENCE_LENGTH,
        pad_value=eng_tokenizer.token_to_id("[PAD]"),
    )
    eng = eng_start_end_packer(eng)

    # Add special tokens (`"[START]"` and `"[END]"`) to `spa` and pad it as well.
    spa_start_end_packer = keras_nlp.layers.StartEndPacker(
        sequence_length=MAX_SEQUENCE_LENGTH + 1,
        start_value=spa_tokenizer.token_to_id("[START]"),
        end_value=spa_tokenizer.token_to_id("[END]"),
        pad_value=spa_tokenizer.token_to_id("[PAD]"),
    )
    spa = spa_start_end_packer(spa)

    return (
        {
            "encoder_inputs": eng,
            "decoder_inputs": spa[:, :-1],
        },
        spa[:, 1:],
    )


def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(2048).prefetch(16).cache()


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

for inputs, targets in train_ds.take(1):
    print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    print(f"targets.shape: {targets.shape}")


# Encoder
encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")

x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=ENG_VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)(encoder_inputs)

encoder_outputs = keras_nlp.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(inputs=x)


encoder_outputs = keras.layers.BatchNormalization()(encoder_outputs)
encoder_outputs = ADAF(N_p = 10, N_m = 10)(encoder_outputs)


encoder = keras.Model(encoder_inputs, encoder_outputs)


# Decoder
decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, EMBED_DIM), name="decoder_state_inputs")

x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=SPA_VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)(decoder_inputs)


x = keras_nlp.layers.TransformerDecoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(decoder_sequence=x, encoder_sequence=encoded_seq_inputs)

x = keras.layers.BatchNormalization()(x)
x = ADAF(N_p = 10, N_m = 10)(x)

 
x = keras.layers.Dropout(0.5)(x)
decoder_outputs = keras.layers.Dense(SPA_VOCAB_SIZE, activation="softmax")(x)


decoder = keras.Model(
    [
        decoder_inputs,
        encoded_seq_inputs,
    ],
    decoder_outputs,
)
decoder_outputs = decoder([decoder_inputs, encoder_outputs])

transformer = keras.Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs,
    name="transformer",
)
encoder.summary()
decoder.summary()
transformer.summary()

def masked_loss(label, pred):
    mask = label != 0
 
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction='none')
    loss = loss_object(label, pred)
 
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss
 
 
def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred
 
    mask = label != 0
 
    match = match & mask
 
    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

optimizer = tf.keras.optimizers.Adam(1e-2, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

transformer.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])

checkpoint_filepath = "./adaf/checkpoint"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_filepath,
    monitor="val_masked_accuracy",
    save_best_only=True,
    save_weights_only=True,
)


history = transformer.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[checkpoint_callback])


loss = np.array(history.history['loss'])
accuracy = np.array(history.history['masked_accuracy'])
val_loss = np.array(history.history['val_loss'])
val_accuracy = np.array(history.history['val_masked_accuracy'])

np.savetxt('./loss_adaf.txt', loss, delimiter=',')
np.savetxt('./accuracy_adaf.txt', accuracy, delimiter=',')
np.savetxt('./val_loss_adaf.txt', val_loss, delimiter=',')
np.savetxt('./val_accuracy_adaf.txt', val_accuracy, delimiter=',')
