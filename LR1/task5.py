import cv2

img = cv2.imread('2.jpg')

cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('HSV Image', cv2.WINDOW_NORMAL)

cv2.imshow('Original Image', img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow('HSV Image', hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()