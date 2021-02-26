import os
import cv2
import math
import numpy as np
from numpy.core.arrayprint import set_printoptions
import pandas as pd
from scipy import fftpack
from matplotlib import pyplot as plt

class CV:
    def __init__(self, vid_path):
        self.video_feed = cv2.VideoCapture(vid_path)

    # Resize the frame
    @staticmethod
    def resize_frame(img, scale_percent=50):
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    def separate_background(self, frame, filter_size=100):
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
        return mat

if __name__ == '__main__':
    # os.system('cls')
    os.system('clear')
    K = np.array([[1406.08415449821,                0, 0],
                  [2.20679787308599, 1417.99930662800, 0],
                  [1014.13643417416, 566.347754321696, 1]])

    cv = CV('Tag1.mp4')
    counter = 0
    while True:
        counter += 1
        ret, frame = cv.video_feed.read()

        if not ret:
            break

        # tag = cv.separate_background(frame, filter_size=220)
        # tag = CV.resize_frame(tag, 50)
        # cv2.imshow("tag", tag)
        # cv2.waitKey(0)

        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to the image (11x11)
        blur = cv2.GaussianBlur(frame, (11,11), 0)

        # Apply Canny edge detector to the image
        edges = cv2.Canny(blur, 200, 250)

        # Apply a closing morphilogical transformation to the image. This will act as a mask
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((61,61), np.uint8))
        
        non0px = np.nonzero(closing)
        med_row = int(np.median(non0px[0]))
        med_col = int(np.median(non0px[1]))

        roi_buffer = 150
        closing[:(med_row-roi_buffer),:] = 0
        closing[(med_row+roi_buffer):, :] = 0

        closing[:, :(med_col-roi_buffer)] = 0
        closing[:, (med_col+roi_buffer):] = 0

        # 
        masked = closing * frame
        masked = cv2.morphologyEx(masked, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        corners = cv2.cornerHarris(masked, 10, 11, .05)
        # frame_cp = masked.copy()
        frame_cp = np.zeros_like(frame)
        
        corners = cv2.dilate(corners, None)
        frame_cp[corners>.01*corners.max()] = 255

        non0px = np.nonzero(frame_cp)
        med_x = np.median(non0px[0])
        med_y = np.median(non0px[1])

        try:
            med_x = int(med_x)
            med_y = int(med_y)
        except ValueError:
            continue

        # roi = np.where(frame)

        min_x_arg = np.min(non0px[0])
        max_x_arg = np.max(non0px[0])
        min_y_arg = np.min(non0px[1])
        max_y_arg = np.max(non0px[1])

        min_dist = min((max_x_arg-min_x_arg, max_y_arg-min_y_arg))
        print(f"frame {counter}: {min_dist}")

        dilate_size = 1
        if min_dist > 250:
            dilate_size = 15
        elif min_dist > 230:
            dilate_size = 13
        elif min_dist > 200:
            dilate_size = 11
        elif min_dist > 170:
            dilate_size = 9
        elif min_dist > 130:
            dilate_size = 7
        elif min_dist > 100:
            dilate_size = 5
        elif min_dist > 75:
            dilate_size = 3

        frame_cp = cv2.dilate(frame_cp, np.ones((dilate_size, dilate_size), dtype=np.uint8))

        # ret, dst = cv.threshold(dst,0.01*dst.max(),255,0)
        # dst = np.uint8(dst)
        # # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(frame_cp)
        # print(f"ret {ret}", f"labels {labels}", f"stats {stats}", f"centroids{centroids}", sep='\n')
        
        corner_filter_1 = None
        num_arr = 5
        while True:
            try:
                corner_filter_1 = np.argpartition(stats[:,-1], num_arr)[:num_arr]
                break
            except:
                num_arr -= 1

        filtered_1 = centroids[corner_filter_1]

        min_x_arg = np.argmin(filtered_1[:,0])
        max_x_arg = np.argmax(filtered_1[:,0])
        min_y_arg = np.argmin(filtered_1[:,1])
        max_y_arg = np.argmax(filtered_1[:,1])

        wait = 0
        try:
            filtered_2 = filtered_1[np.sum(list(range(5)))-(min_x_arg+min_y_arg+max_x_arg+max_y_arg)]
            filtered_2 = (int(round(filtered_2[0])), int(round(filtered_2[1])))
            # print(filtered_2)
            cv2.circle(frame, filtered_2, 5, 0, -1)
        except IndexError:
            print("Index Error")
            # wait = 0

        cv2.circle(frame, (med_y, med_x), 5, 0, -1)
        img_top = np.hstack((frame, closing))
        img_bot = np.hstack((masked, frame_cp))
        img = np.vstack((img_top, img_bot))
        img = CV.resize_frame(img, 30)

        cv2.imshow("img", img)
        k = cv2.waitKey(wait) & 0xff
        if k == ord('q'):
            break
