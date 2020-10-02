import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)

import os
print(os.getcwd())
tf.keras.backend.set_floatx('float64')
# DOWNLOAD DATA
url = './auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()

dataset = dataset.dropna()

# change country to string
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='') #Convert country to number


# Split the data into train and test (80% train / 20% test)
train_dataset = dataset.sample(frac=0.8, random_state=0) # frac sẽ trả về phần trăm trả về(Fraction of axis items to return)
test_dataset = dataset.drop(train_dataset.index) # bỏ những index trong train_dataset đi(Drop index in train_dataset list)
# 
# sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
# plt.show()

# get label
train_features = train_dataset.copy() #value train
test_features = test_dataset.copy() #value test
train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

#The Normalization layer
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

first = np.array(train_features[:1])

# TODO:1 create the horsepower (one input) Normalization (horsepower : mã lực)
horsepower = np.array(train_features['Horsepower'])

horsepower_normalizer = preprocessing.Normalization(input_shape=[1,])
""" input_shape = (1,) chính là kích thước của dữ liệu đầu vào. 
    khi làm việc với dữ liệu nhiều chiều, ta sẽ có các tuple nhiều chiều. 
    Ví dụ, nếu input là ảnh RGB với kích thước 224x224x3 pixel thì input_shape = (224, 224, 3)."""
horsepower_normalizer.adapt(horsepower)

# TODO:2 create multiple input normalizer

dnn_model = tf.keras.Sequential([
    # horsepower_normalizer, #if one varible
    normalizer,
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(units=1)
])

# Giá trị đầu tiên trong Dense bằng 1 thể hiện việc chỉ có 1 unit ở layer này (đầu ra của linear regression trong trường hợp này bằng 1)

#compile => Configures the model for training.
dnn_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss='mean_absolute_error')

# loss built-in loss function see in https://www.tensorflow.org/api_docs/python/tf/keras/losses
# learning rate là tốc độ học càng lớn học càng chậm độ chính xác càng cao (range 0-1)

#use Model.fit() to execute the training
# history = dnn_model.fit(
#     train_features, train_labels, 
#     epochs=100,#số lần training
#     verbose=0, #suppress logging
#     # Calculate validation results on 20% of the training data
#     # validation_split để chia training data thành 2 phần: 80% data sẽ được sử dụng để huấn luyện Model còn 20% còn lại được sử dụng để đánh giá Model
#     validation_split = 0.2)

history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

"""
Giả sử bạn có tập huấn luyện gồm 64.000 images, lựa chọn batch size có giá trị là 64 images.
Đồng nghĩa: mỗi lần cập nhật trọng số, bạn sử dụng 64 images. Khi đó, bạn mất 64.000/64 = 1000 iterations để có thể duyệt qua hết được tập huấn luyện (để hoàn thành 1 epoch).
"""

#view model result
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist)# loss va val_loss càn thấp càn tốt.

#show graph model result
# print(history.history.keys())
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()
# plot_loss(history)

#xem đô chính xác của model với dataset test
test_results = {}
test_results['dnn_model'] = dnn_model.evaluate(
    test_features, test_labels, verbose=0)
print(test_results)

# Save model
# linear_model.save('linear_model')

# Load model
# reloaded = tf.keras.models.load_model('linear_model')

# make predict
print("data predict ---->")
print(test_dataset.tail(5)) #data predict
test_predicts_value = dnn_model.predict(test_features.tail(5)).flatten()
print("result ---->")
print(test_predicts_value) # predict value


# show graph predict and true value
# test_predictions = dnn_model.predict(test_features).flatten()
# a = plt.axes(aspect='equal')
# plt.scatter(test_labels, test_predictions)
# plt.xlabel('True Values [MPG]')
# plt.ylabel('Predictions [MPG]')
# lims = [0, 50]
# plt.xlim(lims)
# plt.ylim(lims)
# _ = plt.plot(lims, lims)
# plt.show()
