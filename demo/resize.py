import cv2

img = cv2.imread("airplane.png")

resized = cv2.resize(img,(300, 300))

cv2.imwrite("airplane1.png",resized)

