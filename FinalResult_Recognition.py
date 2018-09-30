import numpy as np
import cv2
Face_Cascade = cv2.CascadeClassifier('// -->  path of haar cascade of frontalface_alt.xml')
cap = cv2.VideoCapture(0)
rec = cv2.createLBPHFaceRecognizer()
rec.load("Converted_DataSet/MyDataSet_Converted.yml")
Id = 0
Font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
while 1:
    ret,Frame = cap.read()
    gray = cv2.cvtColor(Frame,cv2.COLOR_BGR2GRAY)
    Faces = Face_Cascade.detectMultiScale(gray,1.5,5)
    for (x,y,w,h) in Faces:
        
        cv2.rectangle(Frame,(x,y),(x+w,y+h),(0,0,255),2)
        Id,conf = rec.predict(gray[y:y+h,x:x+w])
        if (Id == 1):
            Id = " Put Your Name "
        else:
            Id = "UnKnown"

        cv2.cv.PutText(cv2.cv.fromarray(Frame),str(Id),(x,y+h),Font,255)
    cv2.imshow("Image",Frame)
    if cv2.waitKey(30) == 27:
        break
cap.release()
cv2.destroyAllWindows()

