import os
import cv2
import math
import numpy as np
from cornerfinder import CornerFinder


class Overlay(CornerFinder):
    def __init__(self) -> None:
        super().__init__()
    
    def get_homography(self, corners, overlayed_corners):
        
        center_square = (np.mean(corners[:,0]), np.mean(corners[:,1]))
        
        def get_quadrant(corner):
            dx = center_square[0] - corner[0]
            dy = center_square[1] - corner[1]
            if dx > 0 and dy > 0:
                return 1
            elif dx < 0 and dy > 0:
                return 2
            elif dx < 0 and dy < 0:
                return 3
            elif dx > 0 and dy > 0:
                return 4
            else:
                return get_quadrant((corner[0]-1, corner[1]-1))
            
        order = [3,4,2,1]
        sorted_corners = []
        for quadrant in order:
            for corner in corners:
                if get_quadrant(corner) == quadrant:
                    sorted_corners.append(corner)
        
        pts = {1: {'x': overlayed_corners[0][0], 'y': overlayed_corners[0][1], 'xp': sorted_corners[0][0], 'yp': sorted_corners[0][1]},
               2: {'x': overlayed_corners[1][0], 'y': overlayed_corners[1][1], 'xp': sorted_corners[1][0], 'yp': sorted_corners[1][1]},
               3: {'x': overlayed_corners[2][0], 'y': overlayed_corners[2][1], 'xp': sorted_corners[2][0], 'yp': sorted_corners[2][1]},
               4: {'x': overlayed_corners[3][0], 'y': overlayed_corners[3][1], 'xp': sorted_corners[3][0], 'yp': sorted_corners[3][1]}}
        
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
    
    def overlay_image(self, frame_color, overlayed):
        
        frame = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
        corners = np.array(self.get_corners(frame))
        # print(corners)
        h,w = overlayed.shape[:2]
        overlayed_corners = np.array([(0,0), (0,h), (w,0), (w,h)])
        try:
            H = self.get_homography(corners, overlayed_corners)
        except:
            return
        
        uv_extrema = []
        for extrema in overlayed_corners:
            pt_temp = H @ np.vstack((extrema[0], extrema[1], 1))
            prop = pt_temp/pt_temp[-1]
            uv_extrema.append((prop[0][0], prop[1][0]))
        
        uv_extrema = np.array(uv_extrema)
        u_min = math.floor(min(uv_extrema[:, 0]))
        v_min = math.floor(min(uv_extrema[:, 1]))
        
        u_max = math.ceil(max(uv_extrema[:, 0]))
        v_max = math.ceil(max(uv_extrema[:, 1]))

        H_inv = np.linalg.inv(H)
        for u in range(u_min, u_max):
            for v in range(v_min, v_max):
                
                xy = H_inv @ np.vstack((u, v, 1))
                x, y = w-1-xy[0]/xy[-1], h-1-xy[1]/xy[-1]
                
                if x < w-1 and x >= 0 and y < h-1 and y >= 0:
                    a = x - math.floor(x)
                    b = y - math.floor(y)
                    
                    S_ij   = overlayed[math.floor(y), math.floor(x)]
                    S_i1j  = overlayed[math.ceil(y), math.floor(x)]
                    S_ij1  = overlayed[math.floor(y), math.ceil(x)]
                    S_i1j1 = overlayed[math.ceil(y), math.ceil(x)]
                        
                    S_i = S_ij + a*(S_ij1 - S_ij)
                    S_j = S_i1j + a*(S_i1j1 - S_i1j)     
                    S = S_i + b*(S_j - S_i)
                    S = [np.uint8(round(S_ii)) for S_ii in S]
                    frame_color[v,u] = S  
                else:
                    continue
                                      
if __name__ == '__main__':
    os.system('cls')
    testudo = cv2.imread("testudo.png")
    overlay = Overlay()
    
    video_feed = cv2.VideoCapture("Tag0.mp4")
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