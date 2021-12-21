# image tracking without machine learning
import cv2
import math
import pandas as pd
import numpy as np


###################################
###################################

def property(roi, cnt):
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)

    solidity = float(area)/hull_area # calculate solidity
    eq = np.sqrt(4*area/np.pi) # calculate equivaletn diameter


    # calculate gray mean value within contour
    mask_array = np.zeros(roi.shape,  np.uint8) # create 0 array having same shape of image
    cv2.drawContours(mask_array, cnt, 255, -1) # within the contour area convert the value as 255 (white)
    masked_img = np.ma.masked_array(roi, mask= (mask_array != 255)) # careful!! 1 is True, 0 is False. in this mask_array, background is False and object is True
    gray_mean = np.mean(masked_img) # unmasked(False=background) will be ignored and masked(Ture=object) will only be considered


    return area, solidity, eq, gray_mean

class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h, roi, cnt = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    #print(self.center_points)
                    properties = property(roi, cnt)
                    objects_bbs_ids.append([x, y, w, h, *properties, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, *properties, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids # return x, y, w, h, area, solidity, eq, gray_mean, id

###################################
###################################
def tracking(filepath):
    tracker = EuclideanDistTracker()

    # reading in video file
    filepath = filepath
    cap = cv2.VideoCapture(filepath)

    # object detection from stable camera
    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=20) # remove the background which does not move at all


    # loooing through the frames
    frame_number = 0
    if "L.mp4" in filepath:
        df = pd.DataFrame(columns=["x", "y", "w", "h", "area", "solidity", "eq", "gray_mean", "obj_id", "frame"])
    elif "R.mp4" in filepath:
        df = pd.DataFrame(columns=["z", "y", "w", "h", "area", "solidity", "eq", "gray_mean", "obj_id", "frame"])

    while True:
        ret, org_frame = cap.read()

        if ret == False:
            df_filepath = filepath.replace(".mp4", ".txt")
            df.to_csv(df_filepath, sep= "\t") # save data as txt file
            print("file saved at %s" %(df_filepath ,))
            break
        # rotate the frame and extract ROI (region of interest)
        if "L.mp4" in filepath:
            frame = cv2.rotate(org_frame, cv2.ROTATE_90_CLOCKWISE)
            roi = frame[300:1200 , 0:900]
        elif "R.mp4" in filepath:
            frame = cv2.rotate(org_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            roi = frame[300:1200 , 0:900]

        # 1. object detection
        # apply mask 
        mask = object_detector.apply(roi) # apply mask on the roi where the background (static object) is removed
        #_, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY) # apply threshold if the object has shadow or so
        contours, hir = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find the countour of moving object

        detections = []
        for cnt in contours:
            # calculater area and remove small elements
            area = cv2.contourArea(cnt)
            if area > 100:
                #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2) # draw line on "roi" with "all" contours with "green line" with thickness 2
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0),  3)

                detections.append([x, y, w, h, roi, cnt])
        
        
        # 2. object tracking
        boxes_ids = tracker.update(detections)
        if len(boxes_ids) == 0: # if there are no objects detected
            df.loc[len(df)] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, int(frame_number)] #########
            frame_number += 1
        else: # if there are objects detected
            for box_id in boxes_ids:
                x, y, w, h, area, solidity, eq, gray_mean, obj_id= box_id
                cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0),  3)

                # append data to dataframe
                df_row = [*box_id, frame_number]
                df.loc[len(df)] = df_row

                frame_number += 1


        
        # show windows
        cv2.imshow("Frame", frame)
        #cv2.imshow("Mask", mask)
        cv2.imshow("ROI", roi)

        
        key = cv2.waitKey(10) # no wait between frames
        if cv2.waitKey(1) & 0xFF == ord('q'): # press "q" to stop streaming
            print(df)
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #filelist = ["sp2_1_L.mp4", "sp2_1_R.mp4", "sp3_2_L.mp4", "sp3_2_R.mp4", "sp4_3_L.mp4", "sp4_3_R.mp4", "sp5_4_L.mp4", "sp5_4_R.mp4",
    #            "sp6_5_L.mp4", "sp6_5_R.mp4", "sp7_1_L.mp4", "sp7_1_R.mp4", "sp8_2_L.mp4", "sp8_2_R.mp4", "sp9_3_L.mp4", "sp9_3_R.mp4",
    #            "sp10_4_L.mp4", "sp10_4_R.mp4", "sp11_5_L.mp4", "sp11_5_R.mp4", "sp12_1_L.mp4", "sp12_1_R.mp4", "sp13_2_L.mp4", "sp13_2_R.mp4",
    #            "sp14_3_L.mp4", "sp14_3_R.mp4", "sp15_4_L.mp4", "sp15_4_R.mp4", "sp16_5_L.mp4", "sp16_5_R.mp4"]
    filelist = ["sp2_1_L.mp4", "sp2_1_R.mp4"]
    for f in filelist:
        tracking("/Users/dkim/Desktop/basler_camera/recording/%s" %(f,))

    

