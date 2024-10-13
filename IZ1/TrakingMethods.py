# выбор область интереса для отслеживания (ROI)
# запускаем код программы и выделяем мышкой интересующую область и наживаем esc
# P.s. так как тестовые видео очень короткие, то для корректного вывода и работы необходимо нажимать "q" до окончания видео

import sys
import time
import cv2

tracker_types = ['CSRT', 'MEDIANFLOW', 'MOSSE']
tracker_type = tracker_types[2]

if tracker_type == 'CSRT':
    tracker = cv2.TrackerCSRT_create()

if tracker_type == 'MEDIANFLOW':
    tracker = cv2.legacy.TrackerMedianFlow_create()

if tracker_type == 'MOSSE':
    tracker = cv2.legacy.TrackerMOSSE_create()

video = cv2.VideoCapture("Car_1.mp4")
ret, frame = video.read()
frame = cv2.resize(frame, (1200, 900))

frame_height, frame_width = frame.shape[:2]
frame = cv2.resize(frame, [frame_width//2, frame_height//2])

output = cv2.VideoWriter('Res'f'_{tracker_type}.avi',
                         cv2.VideoWriter_fourcc(*'XVID'), 60.0,
                         (frame_width//2, frame_height//2), True)
if not ret:
    print('Невозможно прочитать видео')

# Ограничивающая рамка (прямоугольник)
# Выбираем область интереса (ROI) с помощью мыши на первом кадре и инициализируем трекер с заданной областью
bbox = cv2.selectROI(frame, False)
ret = tracker.init(frame, bbox)

# Запоминаем текущее время для измерения времени работы программы
start_time = time.time()

# Запускаем бесконечный цикл для обработки кадров.
# Если кадры закончились, выводим сообщение и выходим из цикла
while True:
    ret, frame = video.read()
    if not ret:
        print('Что-то пошло не так')
        break
    # Уменьшаем размер текущего кадра и запускаем таймер.
    # Обновляем трекер, чтобы получить новые координаты ограничивающего прямоугольника (bbox)
    frame = cv2.resize(frame, [frame_width//2, frame_height//2])
    timer = cv2.getTickCount()
    ret, bbox = tracker.update(frame)
    # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Если обновление трекера прошло успешно, вычисляем координаты верхнего левого (p1)
    # и нижнего правого (p2) углов ограничивающего прямоугольника и рисуем его на кадре.
    if ret:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    # Если трекер не смог обновить координаты, отображаем сообщение о сбое трекинга
    else:
        # Отображаем тип трекера на кадре.
        cv2.putText(frame, "Tracking failure detected", (100, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(frame, tracker_type + " Tracker", (100, 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 170, 50), 2)
    # cv2.putText(frame, "FPS : " + str(int(fps)), (100,50),
    #             cv2.FONT_HERSHEY_DUPLEX, 0.75, (50,170,50),2)

    # Отображаем текущий кадр с отрисованным трекингом и записываем его в выходной файл
    cv2.imshow("Tracking", frame)
    output.write(frame)

    # Ожидаем нажатия клавиши (если нажата клавиша "Esc" - выходим из цикла)
    q = cv2.waitKey(1) & 0xff
    if q == 27:
        break

# Запоминаем время окончания работы программы
end_time = time.time()

# Если в видео есть кадры, выводим время работы трекера и частоту потери кадров
if video.get(cv2.CAP_PROP_FRAME_COUNT) != 0:
    print(f"Время работы метода {tracker_type}: {end_time - start_time:.5f} секунд")
    print(f"Частота потери: {1 / ((end_time - start_time) / video.get(cv2.CAP_PROP_POS_FRAMES)):.0f} fps")

# Освобождаем ресурсы, закрываем видеофайл и все окна OpenCV
video.release()
output.release()
cv2.destroyAllWindows()