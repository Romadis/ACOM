from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("my_model.keras")

image_path ="img/4.jpg"
img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_cv = cv2.resize(img_cv, (28, 28))

# Проверяем изменился ли размер
print(img_cv.shape)

# Нормализуем изображение
image = img_cv / 255.0

# Разворачиваем изображение в одномерный вектор
image = image.reshape(1, 784)

predictions = model.predict(image)

print("Классы: ", predictions)

# Получаем индекс класса с наибольшой вероястностью
predicted_class = np.argmax(predictions)

print("Предсказанный класс:", predicted_class)
