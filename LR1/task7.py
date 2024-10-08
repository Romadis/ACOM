import cv2

def readIPWriteTOFile():
    video = cv2.VideoCapture(0)
    ok, img = video.read()
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("output_2.mov", fourcc, 25, (w, h))
    while (True):
        ok, img = video.read()
        cv2.imshow('Video from webcam', img)
        video_writer.write(img)

        if cv2.waitKey(1) & 0xFF == 27:
            break
            video.release()
            cv2.destroyAllWindows()

readIPWriteTOFile()