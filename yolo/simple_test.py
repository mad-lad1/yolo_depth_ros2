import cv2


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    cv2.imshow("frame", frame)
    cv2.waitKey(1)