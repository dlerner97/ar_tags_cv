import os
import cv2
import math
import numpy as np
import pandas as pd
from scipy import fftpack
from read_ar import ARReader
from matplotlib import pyplot as plt


class CV(ARReader):
    def __init__(self, vid_path):
        super().__init__()
        self.video_feed = cv2.VideoCapture(vid_path)

    # Resize the frame
    @staticmethod
    def resize_frame(img, scale_percent=50):
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    def separate_background_w_fft(self, frame, filter_size=100):
        def disp_fft(img):
            plt.figure()
            plt.imshow(np.log(1+np.abs(img)).real, "gray") 

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        im_fft = fftpack.fft2(frame)
        im_fft_shift = fftpack.fftshift(im_fft)

        r, c = im_fft_shift.shape
        mask = np.zeros_like(frame)
        cv2.circle(mask, (int(c/2), int(r/2)), filter_size, 255, -1)[0]
        high_pass = im_fft_shift * np.invert(mask)/255

        im_fft_ishift = fftpack.ifftshift(high_pass)
        ifft = np.abs(fftpack.ifft2(im_fft_ishift))

        min_val = np.min(ifft)
        max_val = np.max(ifft)
        gray2uint8 = lambda px: np.uint8(255*(px - min_val)/(max_val-min_val))
        mat = np.asarray(list(map(gray2uint8, ifft)))
        _,mat = cv2.threshold(mat, 150, 255, cv2.THRESH_BINARY)
        mat = cv2.dilate(mat, (57,57))
        # mat = cv2.morphologyEx(mat, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        # mat = cv2.morphologyEx(mat, cv2.MORPH_CLOSE, np.ones((31,31), np.uint8))
        return mat

if __name__ == '__main__':
    os.system('cls')
    # os.system('clear')
    K = np.array([[1406.08415449821,                0, 0],
                  [2.20679787308599, 1417.99930662800, 0],
                  [1014.13643417416, 566.347754321696, 1]])

    cv = CV('Tag0.mp4')
    _, frame = cv.video_feed.read()

    tag = cv.separate_background_w_fft(frame, filter_size=30)
    non0 = np.nonzero(tag)

    detected = frame.copy()
    detected[non0[0], non0[1]] = (0,0,255)
    img = np.vstack((frame, detected))

    # tag[non0[0], non0[1]] = 0
    img = CV.resize_frame(img, 20)
    cv2.imshow("all", img)
    cv2.waitKey(0)


