import cv2

img = cv2.imread('1.png')
cv2.namedWindow('Display window', cv2.WINDOW_NORMAL)

#Цвет и толщина фигуры
color = (0, 0, 255)
thickness = 2

height, width, _, = img.shape

#Ширина и высота вертикального прямоугольника
rect_width_1 = 50
rect_height_1 = 400

#Координаты углов
x1_1 = width // 2 - rect_width_1 // 2   #Левый верхний угол по оси x
y1_1 = height // 2 - rect_height_1 // 2 #Левый верхний угол по оси y
x2_1 = width // 2 + rect_width_1 // 2   #Правый нижний угол по оси x
y2_1 = height // 2 + rect_height_1 // 2 #Правый нижний угол по оси y

#Ширина и высота горизонтального прямоугольника
rect_width_2 = 50
rect_height_2 = 350

#Координаты углов
x1_2 = width // 2 - rect_height_2 // 2  #Левый верхний угол по оси x
y1_2 = height // 2 - rect_width_2 // 2  #Левый верхний угол по оси y
x2_2 = width // 2 + rect_height_2 // 2  #Правый нижний угол по оси x
y2_2 = height // 2 + rect_width_2 // 2  #Правый нижний угол по оси y

#Отрисовка
cv2.rectangle(img, (x1_1, y1_1), (x2_1, y2_1), color, thickness)
cv2.rectangle(img, (x1_2, y1_2), (x2_2, y2_2), color, thickness)

#Для размытия центра креста используется GaussianBlur
#Ширина и высота ядра в px для размытия
kernel_size = (5, 15)

#Часть изображения, соответствующая горизонтальному прямоугольнику
img_part = img[y1_2:y2_2, x1_2:x2_2]

img_part_blur = cv2.GaussianBlur(img_part, kernel_size, 30)

#Замена части изображения размытой версией
img[y1_2:y2_2, x1_2:x2_2] = img_part_blur

cv2.imshow('Display window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()