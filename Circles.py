# -*- coding: utf-8 -*-
""" 
PP2 - NN on artificial dataset, binary classification
Two circles, more complicated architecture
Great acc, ~99%-ish
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
np.set_printoptions(suppress=True)

df = make_circles(n_samples = 1000, noise = 0.03)

X = df[0]
Y = df[1]
print(X[0:10, :])
print(Y[0:10])

def create_model_nn():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(4, input_shape=(2,), activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
    return model

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.5)
print(x_train.shape)
nn_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model_nn, epochs=1000, batch_size = 100, 
                                                          callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20),
                                                          validation_data=(x_test, y_test))
history = nn_model.fit(x_train, y_train)
predicted_labels = nn_model.predict(x_test)
print(f"Accuracy is {accuracy_score(predicted_labels, y_test)}")
print(f"Weights are {[l.tolist() for l in nn_model.model.get_weights()]}")