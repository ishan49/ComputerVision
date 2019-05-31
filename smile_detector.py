#smile_detector

import cv2 #library

#creating object of CascadeClassifier class
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('smile_haarcascade.xml')

#defining function that will detect
def detect(grey,frame):
    faces = face_cascade.detectMultiScale(grey,1.3,5) #gives(tuple) the co-ordinates(upper left) with width and height of the rectangles consisting face,image reduced by 1.3 and 5 neighbour zone must be accepted
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #co-ordinate of upper left and lower right point of rectangle and rgb code for color and thickness of rectangle
        roi_grey = grey[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_grey,1.1,5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        smiles = smile_cascade.detectMultiScale(roi_grey,1.7,22)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
    return frame

#using webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read() #two things are retuened by read() by typing '_' we specify that we dont need 1st variable
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas = detect(grey,frame)
    cv2.imshow('Video',canvas) #imshow() shows the images continuosly as video 
    if cv2.waitKey(1) & 0xFF == ord('q'): #when q is pressed in keyboard the face recognition ends
        break
video_capture.release() #turn off webcam
cv2.destroyAllWindows() #destroy window in which images were displayed
