from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt


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


# 네트워크 크기 축소 --> 과대적합 해결 (내부적으로 많은 파라미터 개수가 있어서 저장해야할 게 많음)
smaller_model = models.Sequential()
smaller_model.add(layers.Dense(6, activation = 'relu', input_shape=(10000,)))
smaller_model.add(layers.Dense(6, activation = 'relu'))
smaller_model.add(layers.Dense(1, activation = 'sigmoid'))


smaller_model.compile(optimizer='rmsprop', 
                       loss = 'binary_crossentropy',
                       metrics=['acc'])

original_hist = original_model.fit(x_train, y_train, epochs=20, batch_size=512, validation_data=(x_test, y_test))
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
