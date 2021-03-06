from collections import deque
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video")
ap.add_argument("-b", "--buffer", type=int, default=64)
args = vars(ap.parse_args())

green_lower = (29, 86, 6)
green_upper = (64, 255, 255)
pts = deque(maxlen=args["buffer"])

contour = 0
(dX, dY) = (0, 0)
direction = ""


if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = camera.read()
    if args.get("video") and not grabbed:
        break
    frame = imutils.resize(frame, width=600)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, green_lower, green_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cv2.imshow("mask", mask)

    (_, cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print cnts
    center = None
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            # cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)
            pts.appendleft(center)

    for i in xrange(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue

        if contour >= 10 and i == 1 and len(pts) == args["buffer"] and pts[-10] is not None :
            dX = pts[-10][0] - pts[1][0]
            dY = pts[-10][1] - pts[1][1]
            (dirX, dirY) = ("", "")
            if np.abs(dX) > 20:
                dirX = "East" if np.sign(dX) == 1 else "West"
            if np.abs(dY) > 20:
                dirY = "North" if np.sign(dY) == 1 else "South"
            if dirX != "" and dirY != "":
                direction = "{}-{}".format(dirY, dirX)
            else:
                direction = dirX if dirX != "" else dirY

        thickness = int(np.sqrt(args["buffer"] / float(i+1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0, 0, 255), 3)
    cv2.putText(frame, "dx:{}, dy:{}".format(dX, dY), (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_COMPLEX, 0.35, (0, 0, 255), 1)
    contour += 1
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()