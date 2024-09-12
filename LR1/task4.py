import cv2

#Отображение видео в окне
cap = cv2.VideoCapture(0)
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

#Получение размеров кадра
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#Создание объект VideoWriter для записи видео в файл
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_1.mov', fourcc, 30.0, (width, height))

#Чтение видеофайла кадр за кадром и запись его в другой файл
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        out.write(frame)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()