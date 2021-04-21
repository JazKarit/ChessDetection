# Jaskrit Singh
# CSCI 4831
# Project
# Ioana Fleming 

import cv2

# Convert an mp4 file into frames

cap = cv2.VideoCapture('game4/game4.mp4')

count = 0
while cap.isOpened():
    ret,frame = cap.read()
    cv2.imshow('Game 4', frame)
    cv2.imwrite("game4/frame%d.jpg" % count, frame)
    count = count + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() # destroy all opened windows