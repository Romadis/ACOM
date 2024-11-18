from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# Загрузка данных MNIST и предобработка
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Предобработка изображений и меток (значения пикселей масштабируются от 0 до 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255  # Размерности изменяются для использования сверточных слоев
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

# Преобразование меток в векторы с однократным кодированием
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Создание модели сверточной нейронной сети
model = Sequential()

# Первый сверточный слой с 32 фильтрами
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

# Первый слой пулинга размером 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Второй сверточный слой с 64 фильтрами
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

# Второй слой пулинга размером пула 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Плоский слой, преобразующий данные в одномерный массив перед подачей их на полносвязные слои
model.add(Flatten())

# Полносвязный слой
model.add(Dense(128, activation='relu'))

# Выходной слой
model.add(Dense(10, activation='softmax'))

optimizer = Adam(learning_rate=0.001)

# Компиляция модели
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, batch_size=128, validation_data=(x_test, y_test))

# Оценка модели на тестовых данных
score = model.evaluate(x_test, y_test, verbose=0)

model.save("my_nerone_set.keras")

print('loss: ', score[0])
print('accuracy: ', round(score[1], 3))