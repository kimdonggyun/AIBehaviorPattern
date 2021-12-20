# image tracking without machine learning
import cv2
import math
import pandas as pd


###################################
###################################

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
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    #print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

###################################
###################################

tracker = EuclideanDistTracker()

# reading in video file
filepath = "/Users/dkim/Desktop/basler_camera/recording/sp16_5_L.mp4"
cap = cv2.VideoCapture(filepath)

# object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=20) # remove the background which does not move at all


# loooing through the frames
frame_number = 0
df = pd.DataFrame(columns=["x", "y", "w", "h", "obj_id", "frame"])
while True:
    ret, org_frame = cap.read()
    print(ret)
    # rotate the frame and extract ROI (region of interest)
    if "L.mp4" in filepath:
        frame = cv2.rotate(org_frame, cv2.ROTATE_90_CLOCKWISE)
        roi = frame[350:1200 , 0:900]
    elif "R.mp4" in filepath:
        frame = cv2.rotate(org_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        roi = frame[350:1200 , 0:900]

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
            detections.append([x, y, w, h])

    # 2. object tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0),  3)

        # append data to dataframe
        print(frame_number)
        df_row = box_id.append(frame_number)
        #print(df_row)
        df =df.append(pd.Series(df_row),ignore_index=True)

        frame_number += 1



    # show windows
    cv2.imshow("Frame", frame)
    #cv2.imshow("Mask", mask)
    cv2.imshow("ROI", roi)

    key = cv2.waitKey(10) # no wait between frames
    if cv2.waitKey(1) & 0xFF == ord('q'): # press "q" to stop streaming
        break

print(df)
cap.release()
cv2.destroyAllWindows()


