import cv2

#Флаги для расширения изображений
img1 = cv2.imread('2.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('3.bmp', cv2.IMREAD_COLOR)

#Флаги для чтения изображения
img3 = cv2.imread('1.png', cv2.IMREAD_COLOR)  #Цветное изображение
img4 = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)  #Серое изображение
img5 = cv2.imread('1.png', cv2.IMREAD_UNCHANGED)  #Без изменений

#Флаги для создания окна
cv2.namedWindow('Display window', cv2.WINDOW_NORMAL)
#cv2.namedWindow('Display window', cv2.WINDOW_AUTOSIZE)
#cv2.namedWindow('Display window', cv2.WINDOW_FULLSCREEN)

cv2.imshow('Display window', img4)
cv2.waitKey(0)
cv2.destroyAllWindows()