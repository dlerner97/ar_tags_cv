import os
import cv2
import numpy as np

class ARReader:
    def __init__(self) -> None:
        self.uint8_0 = np.uint8(0)
        self.uint8_half = np.uint8(255//2)    

    def read_tag(self, tag, ret_img=False):
        non0pts = np.nonzero(tag)

        min_non_zero_row = min(non0pts[0])
        max_non_zero_row = max(non0pts[0])
        min_non_zero_col = min(non0pts[1])
        max_non_zero_col = max(non0pts[1])
        
        span_rows = max_non_zero_row - min_non_zero_row
        span_cols = max_non_zero_col - min_non_zero_col
        
        num_bins = 8
        bin_rows = int(round(span_rows/num_bins))
        bin_cols = int(round(span_cols/num_bins))
        
        full_ar_map = np.zeros((num_bins, num_bins), np.uint8)
        
        if not ret_img:
            for i in range(num_bins+1):
                cv2.line(tag, (min_non_zero_col, i*bin_rows), (max_non_zero_col, i*bin_rows), 150, 2)
                cv2.line(tag, (i*bin_cols, min_non_zero_row), (i*bin_cols, max_non_zero_row), 150, 2)
        
        for row in range(num_bins):
            for col in range(num_bins):
                temp_section = tag[row*bin_rows:(row+1)*bin_rows, col*bin_cols: (col+1)*bin_cols]
                full_ar_map[row, col] = np.median(temp_section)
            
        lower = num_bins//4
        upper = lower*3
        qr_tag = full_ar_map[lower:upper, lower:upper]
                        
        get_binary_str = lambda coord: '1' if qr_tag[coord] < self.uint8_half else '0'
        bit_11 = get_binary_str((1,1))
        bit_12 = get_binary_str((1,2))
        bit_21 = get_binary_str((2,1))
        bit_22 = get_binary_str((2,2))
        
        binary_word = {'00'  : int(bit_12+bit_11+bit_21+bit_22,2),
                       '0n1' : int(bit_22+bit_12+bit_11+bit_21,2),
                       'n10' : int(bit_11+bit_21+bit_22+bit_12,2),
                       'n1n1': int(bit_21+bit_22+bit_12+bit_11,2)}
                
        word = None
        if qr_tag[0,0] < self.uint8_half:
            word = binary_word["00"]
        elif qr_tag[0,-1] < self.uint8_half:
             word = binary_word["0n1"]
        elif qr_tag[-1,0] < self.uint8_half:
             word = binary_word["n10"]
        elif qr_tag[-1,-1] <  self.uint8_half:
             word = binary_word["n1n1"]
        else:
            print("Cannot find proper orientation.")     
            
        return word, tag
        
if __name__ == '__main__':
    os.system('cls')
    img = cv2.imread("clean_ar_tag.png")
    # img = cv2.imread("ref_marker2.png")
    # img = cv2.imread("ref_marker3.png")
    # img = cv2.imread("ref_marker4.png")
    reader = ARReader()
    img = 255 - img
    word, tag = reader.read_tag(img)
    print(f"ID: {word}")
    cv2.imshow("Tag w/ grid", tag)
    cv2.waitKey(0)