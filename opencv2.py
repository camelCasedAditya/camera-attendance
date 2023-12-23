import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
#smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

while True:
    ret, frame = cap.read()

    # convert live frame to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # run the preset model and 1.3 = rate that you scale down and 5 = the accuracy of the model.
    faces = face_cascade.detectMultiScale(gray, 1.15, 6)
    #smiles = smile_cascade.detectMultiScale(gray, 1.05, 50)

    #for(sx, sy, sw, sh) in smiles:
        #cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 5)

    # get position of rectangle dimentions and position to draw onto live image
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)

        # find number of pixels in the face rectangle on gray scale
        roi_gray = gray[y:y+w, x:x+w]

        # find number of pixels in the face rectangle on color scale
        roi_color = frame[y:y+h, x:x+w]

        # find location of the eyes on the gray scale face
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.15, 5)
        # get length width and height of eye rectangles
        for (ex, ey, ew, eh) in eyes:

            #draw the eye rectangles onto the color scale face
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 5)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()