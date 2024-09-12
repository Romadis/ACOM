import cv2

#Подключение к IP-камере
cap = cv2.VideoCapture("http://212.192.147.27:8080/video")

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Phone's camera", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        print("Ошибка!")
        break

cap.release()
cv2.destroyAllWindows()