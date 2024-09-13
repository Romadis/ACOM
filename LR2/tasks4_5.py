import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    #Вычисляем момент на основе маски
    moments = cv2.moments(mask)

    #Поиск момента первого порядка
    area = moments['m00']

    if area > 0:
        width = height = int(np.sqrt(area))

        #Вычисляем координаты центра объекта на изображении с использованием момент первого порядка
        c_x = int(moments["m10"] / moments["m00"])
        c_y = int(moments["m01"] / moments["m00"])

        #Отрисовка прямоугольника
        color = (0, 0, 0)
        thickness = 2
        cv2.rectangle(frame,
                      (c_x - (width // 8), c_y - (height // 8)),
                      (c_x + (width // 8), c_y + (height // 8)),
                      color, thickness)

    cv2.imshow('HSV_frame', hsv)
    cv2.imshow('Result_frame', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

print("Площадь объекта:", area)

cap.release()
cv2.destroyAllWindows()
