# image tracking without machine learning
import cv2

# reading in video file
filepath = "/Users/dkim/Desktop/basler_camera/recording/sp16_5_R.mp4"
cap = cv2.VideoCapture(filepath)

# object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2() # remove the background which does not move at all


# loooing through the frames
while True:
    ret, org_frame = cap.read()
    # rotate the frame
    if "L.mp4" in filepath:
        frame = cv2.rotate(org_frame, cv2.ROTATE_90_CLOCKWISE)
    elif "R.mp4" in filepath:
        frame = cv2.rotate(org_frame, cv2.ROTATE_90_COUNTERCLOCKWISE )

    # apply mask 
    mask = object_detector.apply(frame) # apply mask on the frame where the background (static object) is removed
    contours, hir = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find the countour of moving object

    for cnt in contours:
        # calculater area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2) # draw line on "frame" with "all" contours with "green line" with thickness 2

    # show windows
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(10)
    if cv2.waitKey(1) & 0xFF == ord('q'): # press "q" to stop streaming
        break

cap.release()
cv2.destroyAllWindows()


