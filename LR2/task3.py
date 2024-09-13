import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Определяем диапазон красного цвета в HSV
    lower_red = np.array([0, 0, 100])      #Min значения
    upper_red = np.array([100, 100, 255])  #Max значения

    mask = cv2.inRange(hsv, lower_red, upper_red)

    #Применяем маску на изображение
    res = cv2.bitwise_and(frame, frame, mask=mask)

    #Структурирующий элемент, определяющий размер и форму области
    kernel = np.ones((5, 5), np.uint8)

    #Применяем операцию открытия, чтобы удалить шумы и мелкие объекты на изображении
    opening = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)

    #Применяем операцию закрытия, чтобы заполнить маленькие пробелы и разрывы в объектах на изображении
    closing = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('Opening', opening)
    cv2.imshow('Closing', closing)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()