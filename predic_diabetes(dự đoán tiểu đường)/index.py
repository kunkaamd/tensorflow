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
for column in dataset:
    dataset[column] = dataset[column].apply(lambda item: (1 if item == "Yes" else 0) if (item == "Yes" or item == "No") else item )

print(dataset)
