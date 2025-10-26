import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from datetime import datetime
import os

filename = ""

def CaptureVideo():
    cap = cv.VideoCapture(cv.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()

def CaptureFrame():
    global filename
    cap = cv.VideoCapture(cv.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    ret, frame = cap.read()
    filename = './captured_frame' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.jpg'
    cv.imwrite(filename, frame)
    print("Frame captured: " + filename)
    cap.release()
    

def TestHistogram():
    img = cv.imread('test_frame2.png', cv.IMREAD_GRAYSCALE)
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()

def HistogramFromFrame(filename):
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()
    os.remove(filename)
    filename = ""


# TestHistogram()
# CaptureVideo()
CaptureFrame()
HistogramFromFrame(filename)