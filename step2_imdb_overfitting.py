from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers

(train_data, train_labels) , (test_data, test_labels) = imdb.load_data(num_words=10000)

def vectorize_sequences(sequences, dimention=10000):
    results = np.zeros((len(sequences), dimention))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test=np.asarray(test_labels).astype('float32')

original_model = models.Sequential()
original_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
original_model.add(layers.Dense(16, activation='relu'))
original_model.add(layers.Dense(1, activation='sigmoid'))

original_model.compile(optimizer='rmsprop', 
                       loss = 'binary_crossentropy',
                       metrics=['acc'])

original_hist = original_model.fit(x_train, y_train, epochs=20, batch_size=512, validation_data=(x_test, y_test))

"""
# 네트워크 크기 축소 --> 과대적합 해결 (내부적으로 많은 파라미터 개수가 있어서 저장해야할 게 많음)
smaller_model = models.Sequential()
smaller_model.add(layers.Dense(6, activation = 'relu', input_shape=(10000,)))
smaller_model.add(layers.Dense(6, activation = 'relu'))
smaller_model.add(layers.Dense(1, activation = 'sigmoid'))


smaller_model.compile(optimizer='rmsprop', 
                       loss = 'binary_crossentropy',
                       metrics=['acc'])


smaller_hist = smaller_model.fit(x_train, y_train, epochs=20, batch_size=512, validation_data=(x_test, y_test))


epochs = range(1,21)
original_val_loss = original_hist.history['val_loss']
smaller_model_val_loss = smaller_hist.history['val_loss']

plt.plot(epochs, original_val_loss, 'b+', label="Original model")
plt.plot(epochs, smaller_model_val_loss, 'bo', label = 'Smaller model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()


bigger_model = models.Sequential()
bigger_model.add(layers.Dense(1024, activation='relu', input_shape=(10000,)))
bigger_model.add(layers.Dense(1024, activation='relu'))
bigger_model.add(layers.Dense(1, activation='sigmoid'))

bigger_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

bigger_model_hist = bigger_model.fit(x_train, y_train, epochs=20, batch_size=512, validation_data=(x_test, y_test))



bigger_model_val_loss = bigger_model_hist.history['val_loss']
plt.plot(epochs, original_val_loss, 'b+', label="Original model")
plt.plot(epochs, bigger_model_val_loss, 'bo', label = 'Bigger model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()
"""

"""
## 가중치 규제 (keras 에서 regularizers import 로 모델 구성)
l2_model = models.Sequential()
l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu', input_shape=(10000,)))
l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu'))
l2_model.add(layers.Dense(1, activation='sigmoid'))


l2_model.compile(optimizer='rmsprop',
                 loss='binary_crossentropy',
                 metrics=['acc'])

l2_model_hist = l2_model.fit(x_train, y_train,
                             epochs=20,
                             batch_size=512,
                             validation_data=(x_test, y_test))

l2_model_val_loss = l2_model_hist.history['val_loss']



l1_l2_model = models.Sequential()
l1_l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2 = 0.0001),
                          activation='relu', input_shape=(10000,)))
l1_l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l1_l2(l1 = 0.0001, l2 = 0.0001),
                          activation='relu'))
l1_l2_model.add(layers.Dense(1, activation='sigmoid'))


l1_l2_model.compile(optimizer='rmsprop',
                 loss='binary_crossentropy',
                 metrics=['acc'])

l1_l2_model_hist = l1_l2_model.fit(x_train, y_train,
                             epochs=20,
                             batch_size=512,
                             validation_data=(x_test, y_test))
epochs = range(1,21)
original_val_loss = original_hist.history['val_loss']
l1_l2_model_val_loss = l1_l2_model_hist.history['val_loss']


plt.plot(epochs, original_val_loss, 'b', label='Original model')
plt.plot(epochs, l2_model_val_loss, 'b+', label='L1-regularized model')
plt.plot(epochs, l1_l2_model_val_loss, 'bo', label='L1_L2-regularized model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()

"""

# 드롭아웃 추가 : 50 % 의 드롭아웃을 줌
dpt_model = models.Sequential()
dpt_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
dpt_model.add(layers.Dropout(0.5))
dpt_model.add(layers.Dense(16, activation='relu'))
dpt_model.add(layers.Dropout(0.5))
dpt_model.add(layers.Dense(1,activation='sigmoid'))

dpt_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

dpt_model_hist = dpt_model.fit(x_train, y_train, epochs=20, batch_size=51, validation_data=(x_test, y_test))
epochs = range(1,21)
original_val_loss = original_hist.history['val_loss']
dpt_model_val_loss = dpt_model_hist.history['val_loss']


plt.plot(epochs, original_val_loss, 'b', label='Original model')
plt.plot(epochs, dpt_model_val_loss, 'b+', label='drop out model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()
