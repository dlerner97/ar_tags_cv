import os
import cv2
import math
import time
import numpy as np
from numpy.lib.function_base import angle

class CornerFinder:
    def __init__(self) -> None:
        pass
    
    @staticmethod 
    def resize_frame(img, scale_percent=50):
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    def get_corners(self, frame, buffer=10):    
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian Blur to the image (11x11)
        blur = cv2.GaussianBlur(frame, (11,11), 0)
        _, thresholded = cv2.threshold(blur, 230, 255, cv2.THRESH_BINARY)
        structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41,41))
        closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, structure)
        
        num_regions, _, stats, _ = cv2.connectedComponentsWithStats(closed)
        region_filter = np.argpartition(stats[:,cv2.CC_STAT_AREA], num_regions-1) 
        w_region = stats[region_filter[-2], cv2.CC_STAT_WIDTH]
        h_region = stats[region_filter[-2], cv2.CC_STAT_HEIGHT]
        left_region = stats[region_filter[-2], cv2.CC_STAT_LEFT]
        top_region = stats[region_filter[-2], cv2.CC_STAT_TOP]
        
        mask = np.zeros_like(closed)
        mask[top_region-buffer:top_region+h_region+buffer, left_region-buffer:left_region+w_region+buffer] = 255
        closed = cv2.bitwise_and(closed, closed, mask=mask)
        
        inv_thresh = 255 - thresholded
        inner_tag = cv2.bitwise_and(inv_thresh, inv_thresh, mask=closed)
        inner_tag = cv2.morphologyEx(inner_tag, cv2.MORPH_CLOSE, structure)
        
        num_regions, _, stats, _ = cv2.connectedComponentsWithStats(inner_tag)
        region_filter = np.argpartition(stats[:,cv2.CC_STAT_AREA], num_regions-1) 
        w_region = stats[region_filter[-2], cv2.CC_STAT_WIDTH]
        h_region = stats[region_filter[-2], cv2.CC_STAT_HEIGHT]
        left_region = stats[region_filter[-2], cv2.CC_STAT_LEFT]
        top_region = stats[region_filter[-2], cv2.CC_STAT_TOP]
        
        mask = np.zeros_like(closed)
        mask[top_region-buffer:top_region+h_region+buffer, left_region-buffer:left_region+w_region+buffer] = 255
        inner_tag = cv2.bitwise_and(inner_tag, inner_tag, mask=mask)

        corners = cv2.cornerHarris(inner_tag, 10, 11, .05)
        corners = cv2.dilate(corners, None)
        corner_locs = corners>.01*corners.max()
        closed[corner_locs] = 150
        
        corner_img = np.zeros_like(closed, np.uint8)
        corner_img[corner_locs] = 255
        
        num_regions, _, stats, centroids = cv2.connectedComponentsWithStats(corner_img)
        corner_filter_1 = np.argpartition(stats[:, cv2.CC_STAT_AREA], num_regions-1)[:num_regions-1]         
        filtered_1 = centroids[corner_filter_1]

        try:
            final_corners = [(int(round(corner[0])),int(round(corner[1]))) for corner in filtered_1]
            return final_corners, inner_tag
        
            # filtered_corners = np.zeros_like(corner_img, np.uint8)
            # for centroid in final_corners:
            #     cv2.circle(filtered_corners, centroid, 5, 255, -1)

            # cv2.imshow("corners", self.resize_frame(filtered_corners, 80))

            # cv2.imshow("corners", self.resize_frame(filtered_corners, 30))
            # k = cv2.waitKey(1) & 0xff
            # if k == ord('q'):
            #     exit() 
            # elif k == ord('s'):
            #     wait = 
                        
        except TypeError:
            print("Failed")

        return None
        
        
if __name__ == '__main__':
    os.system("cls")
    video_feed = cv2.VideoCapture("Tag0.mp4")
    corner_finder = CornerFinder()
    wait = 1
    try:
        while True:
            ret, frame = video_feed.read()

            if not ret:
                break
            
            frame = corner_finder.resize_frame(frame, 50)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = corner_finder.get_corners(frame)
            
            # img = np.vstack((frame, img))
            img = frame
            
            img = corner_finder.resize_frame(img, 70)
            cv2.imshow("img", img)
            k = cv2.waitKey(wait) & 0xff
            if k == ord('q'):
                break
            elif k == ord('s'):
                if wait == 0:
                    wait = 1
                else:
                    wait = 0
    except KeyboardInterrupt:
        pass
    