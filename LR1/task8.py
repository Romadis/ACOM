import cv2
import numpy as np

img = cv2.imread('1.png')
cv2.namedWindow('Display window', cv2.WINDOW_NORMAL)

#Цвет и толщина прямоугольников и линии
color = (0, 0, 255)
thickness = 2

height, width, _, = img.shape

#Ширина и высота вертикального прямоугольника
rect_width_1 = 50
rect_height_1 = 400

#Координаты углов
x1_1 = width // 2 - rect_width_1 // 2    #Левый верхний угол по оси x
y1_1 = height // 2 - rect_height_1 // 2  #Левый верхний угол по оси y
x2_1 = width // 2 + rect_width_1 // 2    #Правый нижний угол по оси x
y2_1 = height // 2 + rect_height_1 // 2  #Правый нижний угол по оси y

#Ширина и высота горизонтального прямоугольника
rect_width_2 = 50
rect_height_2 = 350

#Координаты углов
x1_2 = width // 2 - rect_height_2 // 2  #Левый верхний угол по оси x
y1_2 = height // 2 - rect_width_2 // 2  #Левый верхний угол по оси y
x2_2 = width // 2 + rect_height_2 // 2  #Правый нижний угол по оси x
y2_2 = height // 2 + rect_width_2 // 2  #Правый нижний угол по оси y

#Отрисовка
cv2.rectangle(img, (x1_1-5, y1_1-3), (x2_1-5, y2_1-3), color, thickness)
cv2.rectangle(img, (x1_2-5, y1_2-3), (x2_2-5, y2_2-3), color, thickness)

#Для размытия центра креста используется GaussianBlur
#Ширина и высота ядра в px для размытия
kernel_size = (71, 11)

#Часть изображения, соответствующая горизонтальному прямоугольнику
img_part = img[y1_2:y2_2, x1_2:x2_2]

img_part_blur = cv2.GaussianBlur(img_part, kernel_size, 30)

#Замена части изображения размытой версией
img[y1_2:y2_2, x1_2:x2_2] = img_part_blur

#Получение цвета центрального пикселя в формате RGB
cx = width // 2        #Координата x центра изображения
cy = height // 2       #Координата y центра изображения
r, g, b = img[cy][cx]  #Компоненты цвета в RGB

#Определение ближайшего цвета
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  #Список возможных цветов в формате RGB
distances = []                                    #Список расстояний от центрального пикселя до каждого цвета

#Расстояние между 2-мя цветами в RGB вычисляем по Евклидовому расстоянию между их координатами
for color in colors:
    distance = np.sqrt((r - color[0])**2 + (g - color[1])**2 + (b - color[2])**2)
    distances.append(distance)

#Индекс ближайшего цвета в списке colors соответствует минимальному расстоянию в списке distances
min_index = distances.index(min(distances))

#Выбор ближайшего цвета из списка colors по индексу
nearest_color = colors[min_index]

#Закрашивание креста ближайшим цветом
cv2.rectangle(img, (x1_1-5, y1_1-3), (x2_1-5, y2_1-3), nearest_color, -1)   #-1 заполнение всей области
cv2.rectangle(img, (x1_2-5, y1_2-3), (x2_2-5, y2_2-3), nearest_color, -1)

cv2.imshow('Display window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
