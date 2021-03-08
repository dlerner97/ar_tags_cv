import os
import cv2
import math
import numpy as np
from cornerfinder import CornerFinder
from read_ar import ARReader
import matplotlib.pyplot as plt


class Overlay2D(CornerFinder, ARReader):
    def __init__(self) -> None:
        CornerFinder.__init__(self)
        ARReader.__init__(self)
        np.seterr(all='raise')
    
    def get_homography(self, corners, overlaid_corners, start_sort=1,order=[3,4,2,1]):
        
        sort_dict = {1: [1,3,4,2],
                     2: [2,1,3,4],
                     3: [3,4,2,1],
                     4: [4,2,1,3]}
        order = sort_dict[start_sort]
        
        center_square = (np.mean(corners[:,0]), np.mean(corners[:,1]))
        overlaid_corners = np.array(overlaid_corners)
        center_overlaid = (np.mean(overlaid_corners[:,0]), np.mean(overlaid_corners[:,1]))
        
        def get_quadrant(corner, center):
            dx = corner[0] - center[0]
            dy = corner[1] - center[1]
            
            if dx > 0 and dy > 0:
                return 1
            elif dx < 0 and dy > 0:
                return 2
            elif dx < 0 and dy < 0:
                return 3
            elif dx > 0 and dy < 0:
                return 4
            else:
                return get_quadrant((corner[0]-1, corner[1]-1), center)
        
        sorted_corners = []
        for quadrant in order:
            for corner in corners:
                if get_quadrant(corner, center_square) == quadrant:
                    sorted_corners.append(corner)
                    break

        sorted_overlaid = []
        for quadrant in order:
            for corner in overlaid_corners:
                if get_quadrant(corner, center_overlaid) == quadrant:
                    sorted_overlaid.append(corner)
                    break           
                     
        pts = {1: {'x': sorted_overlaid[0][0], 'y': sorted_overlaid[0][1], 'xp': sorted_corners[0][0], 'yp': sorted_corners[0][1]},
               2: {'x': sorted_overlaid[1][0], 'y': sorted_overlaid[1][1], 'xp': sorted_corners[1][0], 'yp': sorted_corners[1][1]},
               3: {'x': sorted_overlaid[2][0], 'y': sorted_overlaid[2][1], 'xp': sorted_corners[2][0], 'yp': sorted_corners[2][1]},
               4: {'x': sorted_overlaid[3][0], 'y': sorted_overlaid[3][1], 'xp': sorted_corners[3][0], 'yp': sorted_corners[3][1]}}
        
        A = [[-pts[1]['x'], -pts[1]['y'], -1, 0, 0, 0, pts[1]['x']*pts[1]['xp'], pts[1]['y']*pts[1]['xp'], pts[1]['xp']],
             [0, 0, 0, -pts[1]['x'], -pts[1]['y'], -1, pts[1]['x']*pts[1]['yp'], pts[1]['y']*pts[1]['yp'], pts[1]['yp']],
             [-pts[2]['x'], -pts[2]['y'], -1, 0, 0, 0, pts[2]['x']*pts[2]['xp'], pts[2]['y']*pts[2]['xp'], pts[2]['xp']],
             [0, 0, 0, -pts[2]['x'], -pts[2]['y'], -1, pts[2]['x']*pts[2]['yp'], pts[2]['y']*pts[2]['yp'], pts[2]['yp']],
             [-pts[3]['x'], -pts[3]['y'], -1, 0, 0, 0, pts[3]['x']*pts[3]['xp'], pts[3]['y']*pts[3]['xp'], pts[3]['xp']],
             [0, 0, 0, -pts[3]['x'], -pts[3]['y'], -1, pts[3]['x']*pts[3]['yp'], pts[3]['y']*pts[3]['yp'], pts[3]['yp']],
             [-pts[4]['x'], -pts[4]['y'], -1, 0, 0, 0, pts[4]['x']*pts[4]['xp'], pts[4]['y']*pts[4]['xp'], pts[4]['xp']],
             [0, 0, 0, -pts[4]['x'], -pts[4]['y'], -1, pts[4]['x']*pts[4]['yp'], pts[4]['y']*pts[4]['yp'], pts[4]['yp']]]
               
        _, _, V = np.linalg.svd(A)
        H = V[-1, :]
        H = (H/H[-1]).reshape(3,3)
        return H
    
    def apply_homography(self, frame_color, H, overlaid, overlaid_corners):
        h,w = overlaid.shape[:2]
        s_frame = frame_color.shape

        grayscale = True
        if len(s_frame) == 3:
            grayscale = False

        uv_extrema = []
        for extrema in overlaid_corners:
            pt_temp = H @ np.vstack((extrema[0], extrema[1], 1))
            prop = pt_temp/pt_temp[-1]
            uv_extrema.append((prop[0][0], prop[1][0]))
                    
        # print(f"s_frame: {s_frame}")
        # print(f"overlaid: h: {h}, w: {w}")   
        # print(f"overlaid: \n{overlaid_corners}")
        # print(np.array(uv_extrema))         
                    
        uv_extrema = np.array(uv_extrema)
        u_min = math.floor(min(uv_extrema[:, 0]))
        v_min = math.floor(min(uv_extrema[:, 1]))
        
        u_max = math.ceil(max(uv_extrema[:, 0]))
        v_max = math.ceil(max(uv_extrema[:, 1]))

        H_inv = np.linalg.inv(H)
        for u in range(u_min, u_max):
            for v in range(v_min, v_max):
                
                xy = H_inv @ np.vstack((u, v, 1))
                if grayscale:
                    x, y = xy[0]/xy[-1], xy[1]/xy[-1]
                else:
                    x, y = xy[0]/xy[-1], xy[1]/xy[-1]
                
                if x < w-1 and x >= 0 and y < h-1 and y >= 0:
                    a = x - math.floor(x)
                    b = y - math.floor(y)
                    
                    S_ij   = overlaid[math.floor(y), math.floor(x)]
                    S_i1j  = overlaid[math.ceil(y), math.floor(x)]
                    S_ij1  = overlaid[math.floor(y), math.ceil(x)]
                    S_i1j1 = overlaid[math.ceil(y), math.ceil(x)]
                        
                    try:  
                        S_i = S_ij + a*(S_ij1 - S_ij)
                    except FloatingPointError:
                        S_i = 255
                    
                    try:
                        S_j = S_i1j + a*(S_i1j1 - S_i1j)
                    except:
                        S_j = 255
                        
                    try:
                        S = S_i + b*(S_j - S_i)
                    except:
                        S = 255
                    
                    S = [np.uint8(round(S_ii)) for S_ii in S]
                    
                    # print("-------------------")
                    # print(f"x: {x}, y: {y}")
                    # print(f"u: {u}, v: {v}")
                    # print(f"S_ij: {S_ij}, S_i1j: {S_i1j}, S_ij1: {S_ij1}, S_i1j1: {S_i1j1}")
                    # print(f"S_i: {S_i}, S_j: {S_j}, S: {S}")
                        
                    if not grayscale:
                        frame_color[v,u] = S
                    elif v >= 0 and u >= 0 and u < s_frame[1] and v < s_frame[0]:
                        frame_color[v,u] = 255 if S[0] > 200 else 0
                            
                else:
                    continue
        # plt.imshow(frame_color, 'gray')
        # plt.figure()
        # plt.imshow(overlaid, 'gray')
        # plt.show()
        
        return frame_color
    
    def overlay_image(self, frame_color, overlaid):
        
        frame = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
        corners, inner_tag_mask = self.get_corners(frame)
        corners = np.array(corners)
        inner_tag = cv2.bitwise_and(frame, frame, mask=inner_tag_mask)        
        
        sz_grid = 200
        square = np.zeros((sz_grid,sz_grid), np.uint8)
        square_corners = np.array([(0,0), (sz_grid,0), (0,sz_grid), (sz_grid,sz_grid)])
        
        try:
            H_tag = self.get_homography(square_corners, corners)
        except:
            pass        
        grid = self.apply_homography(square, H_tag, inner_tag, corners)
        _, quadrant = self.read_tag(grid)
        
        h,w = overlaid.shape[:2]
        overlaid_corners = np.array([(0,0), (0,h), (w,0), (w,h)])
        try:
            H = self.get_homography(corners, overlaid_corners, quadrant)
        except:
            return
        
        self.apply_homography(frame_color, H, overlaid, overlaid_corners)
                                      
if __name__ == '__main__':
    from os import system
    system('cls')
    testudo = cv2.imread("testudo.png")
    overlay = Overlay2D()
    
    video_feed = cv2.VideoCapture("Tag1.mp4")
    wait = 0
    try:
        while True:
            ret, frame = video_feed.read()

            if not ret:
                break
            
            frame = overlay.resize_frame(frame, 50)
            frame_cp = frame.copy()
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            overlay.overlay_image(frame, testudo)

            # img = np.vstack((frame_cp, frame))
            img = frame
            
            # img = overlay.resize_frame(img, 80)
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