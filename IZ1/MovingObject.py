# Код выполняет обработку видео, используя метод вычитания фона для отслеживания движущихся объектов

import cv2
import time

# Зпгружаем исходный видеофайл (вручную меняем имя файла для тестирования остальных)
cap = cv2.VideoCapture('Car_1.mp4')

# Инициализируем трекер CSRT
fgbg = cv2.createBackgroundSubtractorMOG2()

# Читаем 1-ый кадр из видеофайла
ret, frame = cap.read()

# Получаем ширину и высоту кадра
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Создаем объект cv2.VideoWriter (Кодек)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))

start_time = time.time()

# Читаем видеопоток и отслеживаем объекты
while True:
    # Читаем кадр из видеофайла
    ret, frame = cap.read()

    # Обновляем трекер на текущем кадре
    fgmask = fgbg.apply(frame)

    # Ищем контуры на маске переднего плана
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Перебираем контуры
    for contour in contours:
        # Получаем ограничивающий прямоугольник каждого контура
        x, y, w, h = cv2.boundingRect(contour)

        # Отрисовываем прямоугольник вокруг контура
        if w > 50 and h > 50 and w/h > 1.3:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Записываем текущий кадр в файл
    out.write(frame)

    # Отображаем текущий кадр
    cv2.imshow('Tracking', frame)

    # Выходим из цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
end_time = time.time()

# Выводим сравнительные характеристии
if cap.get(cv2.CAP_PROP_FRAME_COUNT) != 0:
    print(f"Время работы метода CSRT: {end_time - start_time:.5f} сек")
    print(f"Скорость обработки: {cap.get(cv2.CAP_PROP_FPS):.0f} fps")
    print(f"Частота потери изображения: {1 / ((end_time - start_time) / cap.get(cv2.CAP_PROP_POS_FRAMES)):.0f} fps")
else:
    print("Видеофайл не содержит кадров!")

cap.release()
out.release()
cv2.destroyAllWindows()
