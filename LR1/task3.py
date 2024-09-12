import cv2

#Отображение видео в окне
cap = cv2.VideoCapture('Video1.mp4', cv2.WINDOW_NORMAL)
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

#Изменение размера окна
cv2.resizeWindow('Video', 800, 600)
#cv2.resizeWindow('Video', 1024, 1000)
#cv2.resizeWindow('Video', 1800, 800)

#Чтение видеофайла кадр за кадром
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        #Изменение цветовой гаммы кадра
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vsh = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        #Отображение кадра в окне
        cv2.imshow('Video', gray)

        #Выход при нажатии 'esc'
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()