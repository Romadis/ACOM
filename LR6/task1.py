import tensorflow as tf
import numpy as np
import ssl
import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard

ssl._create_default_https_context = ssl._create_unverified_context

# Загружаем данные train и test
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Масштабируем значения пикселя в пределах [0;1] (для ускорения обучения)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Изменяем форму изображений из 2D (28x28) в 1D (784 элемента)
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

# Преобразуем метки в векторы с однократным кодированием, чтобы сеть работала с многоклассовой классификацией
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Создаем последовательную модель, в которой слои будут добавляться один за другим
model = Sequential()

# 2 скрытых слоя (512 и 768 единиц измерения)
model.add(Dense(512, input_shape=(784, ), activation='relu'))
model.add(Dense(768, activation='relu'))

# Выходной слой
model.add(Dense(10, activation='softmax'))

optimizer = Adam(learning_rate=0.001)

# Компилируем модель с помощью оптимизатора Adam и категориальной функции потери перекрестной энтропии
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Обновляем веса модели после каждых 128 образцов
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
model.fit(x_train, y_train, epochs=3, batch_size=128, callbacks=[tensorboard_callback])

# Оценка точности модели на основе тестовых данных
accuracy = model.evaluate(x_test, y_test)
print(f'Точность модели = {accuracy[1]*100:.2f}%')

model.save("my_model.keras")