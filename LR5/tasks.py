import cv2

i = 0

def main(kernel_size, standard_deviation, delta_tresh, min_area):
    global i
    i += 1

    video = cv2.VideoCapture('1.mov', cv2.CAP_ANY)

    ret, frame = video.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_writer = cv2.VideoWriter('result' + str(i) + '.mp4', fourcc, 25, (w, h))

    while True:
        print('Обрабатываю видео...')

        old_img = img.copy()

        is_ok, frame = video.read()
        if not is_ok:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

        frame_diff = cv2.absdiff(img, old_img)

        thresh = cv2.threshold(frame_diff, delta_tresh, 255, cv2.THRESH_BINARY)[1]

        (contors, hierarchy) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contr in contors:
            area = cv2.contourArea(contr)
            if area < min_area:
                continue
            video_writer.write(frame)
    video_writer.release()
    print('Обработка завершена, УРА!')


kernel_size = 3
standard_deviation = 50
delta_tresh = 60
min_area = 20
main(kernel_size, standard_deviation, delta_tresh, min_area)

kernel_size = 3
standard_deviation = 50
delta_tresh = 40
min_area = 5
main(kernel_size, standard_deviation, delta_tresh, min_area)

kernel_size = 5
standard_deviation = 50
delta_tresh = 70
min_area = 30
main(kernel_size, standard_deviation, delta_tresh, min_area)

kernel_size = 5
standard_deviation = 30
delta_tresh = 50
min_area = 15
main(kernel_size, standard_deviation, delta_tresh, min_area)

kernel_size = 11
standard_deviation = 70
delta_tresh = 60
min_area = 20
main(kernel_size, standard_deviation, delta_tresh, min_area)




