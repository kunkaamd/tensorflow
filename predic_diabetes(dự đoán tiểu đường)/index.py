# CHOOSE LOSS FUCNTION
# -Regression Loss Functions
# Mean Squared Error Loss
# Mean Squared Logarithmic Error Loss
# Mean Absolute Error Loss

# -Binary Classification Loss Functions
# Binary Cross-Entropy
# Hinge Loss
# Squared Hinge Loss

# -Multi-Class Classification Loss Functions
# Multi-Class Cross-Entropy Loss
# Sparse Multiclass Cross-Entropy Loss
# Kullback Leibler Divergence Loss

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
# load dataset
url = './diabetes_data_upload.csv'
raw_dataset = pd.read_csv(url)
dataset = raw_dataset.copy()
dataset = dataset.dropna()

# convert data set to int (Yes,No => 1,0)
def convertDataToInt(item):
    if item == "Yes":
        return 1
    elif item == "No":
        return 0
    elif item == "Male":
        return 1
    elif item == "Female":
        return 0
    elif item == "Negative":
        return 0
    elif item == "Positive":
        return 1
    else:
        return item

for column in dataset:
    dataset[column] = dataset[column].apply(convertDataToInt)

train_dataset = dataset.sample(frac=0.8, random_state=0) # frac sẽ trả về phần trăm trả về(Fraction of axis items to return)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy() #value train
test_features = test_dataset.copy() #value test
train_labels = train_features.pop('class')
test_labels = test_features.pop('class')

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features,dtype=np.int16))

model = keras.Sequential([
    normalizer,
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(units=1,activation='sigmoid')
])
print(model.output_shape)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_features, train_labels, epochs=25)

print("___history___")
print(history)

test_loss, test_acc = model.evaluate(test_features,  test_labels, verbose=0)
"""
Make predict with classes
* `np.argmax(model.predict(x), axis=-1)`,   if your model does multi-class classification   (e.g. if it uses a `softmax` last-layer activation).
* `(model.predict(x) > 0.5).astype("int32")`,   if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).
"""
predict = model.predict(test_features.tail(7))
predict_class = (predict > 0.5).astype("int32")
print("Predict Label:",predict_class)
print("True Label",test_labels.tail(7))

