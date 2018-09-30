import numpy as np
import cv2
Face_Cascade = cv2.CascadeClassifier('// -->  path of haar cascade of frontalface_alt.xml')
cap = cv2.VideoCapture(0)
cv2.namedWindow("Get Image For User",cv2.WINDOW_AUTOSIZE)
while 1:
    ret,Img = cap.read()
    gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    Faces = Face_Cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in Faces:
        cv2.imwrite("MyDataSet/User."+str(id)+ "." +str(NumberOf_Face_For_One_User)+ ".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(Img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.waitKey(100)
    cv2.imshow("Get Image For User",Img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
    
