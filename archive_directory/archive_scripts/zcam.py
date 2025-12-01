import cv2
import numpy as np
import matplotlib.pyplot as plt



count = 0
cap = cv2.VideoCapture(0)

if(cap.isOpened()==False):
    print("ERROR")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow("Frame", frame)
        name = "frame%d.jpg" % count
        cv2.imwrite(name, frame)
        count +=1

    if cv2.waitKey(1):
        break

img = cv2.imread("frame0.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(121), plt.imshow(gray, cmap='gray'), plt.axis("off"), plt.title("Grayscale")
plt.subplot(122), plt.hist(gray.ravel(),256,[0,256],color='k'), plt.title("Gray Histogram")
plt.show()


cap.release()
cv2.destroyAllWindows()