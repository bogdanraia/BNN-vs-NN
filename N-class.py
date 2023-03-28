# -*- coding: utf-8 -*-
""" 
PP2 - NN on artificial dataset, multiclass classification
Synthetic N-class, more complicated architecture
Good acc, ~90%-ish
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
np.set_printoptions(suppress=True)

df = make_classification(n_samples = 1000, n_features = 3, n_informative = 3, n_redundant = 0, n_classes=4, n_clusters_per_class = 1, class_sep=1.5)
print(df)

X = df[0]
Y = df[1]
print(X[0:10, :])
print(Y[0:10])

def create_model_nn():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(6, input_shape=(3,), activation='relu'))
    model.add(tf.keras.layers.Dense(6, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
    return model

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.5)
print(x_train.shape)
nn_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model_nn, epochs=1000, batch_size = 500, 
                                                          callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20),
                                                          validation_data=(x_test, y_test))
history = nn_model.fit(x_train, y_train)
predicted_labels = nn_model.predict(x_test)
print(f"Accuracy is {accuracy_score(predicted_labels, y_test)}")
print(f"Weights are {[l.tolist() for l in nn_model.model.get_weights()]}")