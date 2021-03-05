        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to the image (11x11)
        blur = cv2.GaussianBlur(frame, (11,11), 0)

        # Apply Canny edge detector to the image
        edges = cv2.Canny(blur, 150, 250)
        cv2.imshow("edge", edges)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours))
        print(hierarchy[0])

        # Draw contours
        drawing = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
            print(contours[i])
            perim = cv2.arcLength(cv2.moments(contours[i]), True)
            print(perim)
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
        # Show in a window
        cv2.imshow('Contours', drawing)


        non0px = np.nonzero(edges)
        med_row = int(np.median(non0px[0]))
        med_col = int(np.median(non0px[1]))

        roi_buffer = 100
        edges[:(med_row-roi_buffer),:] = 0
        edges[(med_row+roi_buffer):, :] = 0

        edges[:, :(med_col-roi_buffer)] = 0
        edges[:, (med_col+roi_buffer):] = 0

        # Apply a closing morphilogical transformation to the image. This will act as a mask
        # edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((11,11), np.uint8))
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((61,61), np.uint8))
        cv2.imshow("edges",edges)
        
        non0px = np.nonzero(closing)
        med_row = int(np.median(non0px[0]))
        med_col = int(np.median(non0px[1]))
        
        # 
        masked = closing * frame
        
        masked = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, np.ones((51,51), np.uint8))
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
        num_arr = ret-1
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

        c1 = (int(round(centroids[min_x_arg,0])), int(round(centroids[min_x_arg,1])))
        c2 = (int(round(centroids[max_x_arg,0])), int(round(centroids[max_x_arg,1])))
        c3 = (int(round(centroids[min_y_arg,0])), int(round(centroids[min_y_arg,1])))
        c4 = (int(round(centroids[max_y_arg,0])), int(round(centroids[max_y_arg,1])))

        cv2.circle(frame, c1, 5, 0, -1)
        cv2.circle(frame, c2, 5, 0, -1)
        cv2.circle(frame, c3, 5, 0, -1)
        cv2.circle(frame, c4, 5, 0, -1)

        # cv2.circle(frame, (med_y, med_x), 5, 0, -1)
        img_top = np.hstack((frame, closing))
        img_bot = np.hstack((masked, frame_cp))
        img = np.vstack((img_top, img_bot))
        img = CV.resize_frame(img, 30)
        # plt.imshow(CV.resize_frame(frame,30))
        wait = 0
        cv2.imshow("img", img)
        k = cv2.waitKey(wait) & 0xff
        if k == ord('q'):
            break