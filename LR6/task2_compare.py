import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import time

model = load_model("my_model.keras")

epochs_list = [1, 2, 3]
learning_rates = [0.001, 0.01, 0.1]
num_layers_list = [1, 2, 3]
accuracies = []

# Загружаем и предобрабатываем данные MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Преобразовываем изображения в одномерные массивы
train_images_flat = train_images.reshape((60000, 784))
test_images_flat = test_images.reshape((10000, 784))

train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Перебираем различные значения: эпох, скоростей обучения и количества слоёв
for epochs in epochs_list:
    for lr in learning_rates:
        for num_layers in num_layers_list:
            # Создание модели с определенным количеством слоев
            layered_model = tf.keras.Sequential()
            for _ in range(num_layers):
                layered_model.add(tf.keras.layers.Dense(128, activation='relu'))

            layered_model.add(tf.keras.layers.Dense(10, activation='softmax'))

            layered_model.compile(optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=lr),
                                  loss='categorical_crossentropy', metrics=['accuracy'])

            # Обучение модели
            start_time = time.time()
            layered_model.fit(train_images_flat, train_labels, epochs=epochs, batch_size=128, verbose=0)
            end_time = time.time()
            test_time = end_time - start_time

            # Оценка модели на тестовых данных
            test_loss, test_accuracy = layered_model.evaluate(test_images_flat, test_labels, verbose=0)

            # Добавление параметров в список
            accuracies.append((epochs, lr, num_layers, test_accuracy, test_time))

for epochs, lr, num_layers, accuracy, t_time in accuracies:
    print(f'Эпохи: {epochs}, '
          f'Скорость обучения: {lr}, '
          f'Количество слоев: {num_layers}, '
          f'Точность на тесте: {round(accuracy, 2) * 100}%, '
          f'Затраченное время: {round(t_time, 2)} сек.'
          )
