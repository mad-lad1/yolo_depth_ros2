#program to write a video capture into a file using opencv. Use MP4 files

import cv2
import datetime
import os
cap = cv2.VideoCapture(0)
# video codec h264
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
# get the time and date as a string
now = datetime.datetime.now()
# write into a an mp4 file
# create a folder called capture and save the file there but go back one dir
path = os.path.join(os.path.dirname(__file__), 'capture')
if not os.path.exists(path):
    os.mkdir(path)
# create a file name with the dddate and time
file_name = now.strftime("%Y-%m-%d_%H-%M-%S") + '.avi'
#= create the full path to the file
file_path = os.path.join(path, file_name)
# create the video writer object
video_writer = cv2.VideoWriter(file_path, fourcc, 20.0, (1920, 1080))


while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        # resize the frame to HD 
        frame = cv2.resize(frame, (1920, 1080))
        video_writer.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
video_writer.release()
cv2.destroyAllWindows()