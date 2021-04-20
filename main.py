#Initializing the libraries
import cv2
import os

#Cascade Classifiers and Haar Features are the methods used for Object Detection
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
face = cv2.CascadeClassifier(cascPath)


capture_video = cv2.VideoCapture(0)

while True:
    #Capturing video frame by frame
    ret, frame = capture_video.read()

    color = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(color, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    #Drawing rectange around faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    #Show the output
    cv2.imshow('Video', frame)
    #If pressed then exit capturing video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture_video.release()
cv2.destroyAllWindows()