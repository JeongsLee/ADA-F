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
        inputs = inputs +self.L/2.
        return self.out_g_x_3(inputs)+tf.math.multiply(self.beta*self.out_g_x_2(inputs),inputs) 


# One-hot / categorical encoding
def input_preprocess(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False
       
    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)    

    x = layers.BatchNormalization()(x)    
    x = ADAF(N_p = 5, N_m = 5)(x) 
    
    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

# IMG_SIZE is determined by EfficientNet model choice
IMG_SIZE = 224

batch_size = 64

dataset_name = "stanford_dogs"
(ds_train, ds_test), ds_info = tfds.load(
    dataset_name, split=["train", "test"], with_info=True, as_supervised=True
)
NUM_CLASSES = ds_info.features["label"].num_classes

size = (IMG_SIZE, IMG_SIZE)
ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))



img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


ds_train = ds_train.map(
    input_preprocess, num_parallel_calls=tf.data.AUTOTUNE
)
ds_train = ds_train.batch(batch_size=batch_size, drop_remainder=True)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(input_preprocess)
ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)


checkpoint_filepath = "./adaf/checkpoint"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_filepath,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
)

model = build_model(num_classes=NUM_CLASSES)
model.summary()
epochs = 40  # @param {type: "slider", min:8, max:80}
history = model.fit(ds_train, epochs=epochs, validation_data=ds_test, callbacks=[checkpoint_callback])

loss = np.array(history.history['loss'])
accuracy = np.array(history.history['accuracy'])
val_loss = np.array(history.history['val_loss'])
val_accuracy = np.array(history.history['val_accuracy'])

np.savetxt('./loss_adaf.txt', loss, delimiter=',')
np.savetxt('./accuracy_adaf.txt', accuracy, delimiter=',')
np.savetxt('./val_loss_adaf.txt', val_loss, delimiter=',')
np.savetxt('./val_accuracy_adaf.txt', val_accuracy, delimiter=',') 

model.save_weights('./checkpoints/adaf/ckpoint_adaf')