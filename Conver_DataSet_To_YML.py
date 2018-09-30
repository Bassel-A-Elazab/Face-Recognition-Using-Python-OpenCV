import os
import numpy as np 
import cv2
from PIL import Image
recognizer = cv2.createLBPHFaceRecognizer()
Path = "Final Face Recognition\\MyDataSet"

def getImageWithId(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    Faces = []
    IDs = []
    for imagePath in imagePaths:
        FaceImage = Image.open(imagePath).convert("L")
        FaceNP = np.array(FaceImage,"uint8")
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        Faces.append(FaceNP)
        IDs.append(ID)
        cv2.imshow("Convert My DataSet",FaceNP)
        cv2.waitKey(10)
    return np.array(IDs),Faces
IDs , Faces = getImageWithId(Path)
recognizer.train(Faces,IDs)
recognizer.save("Converted_DataSet\\MyDataSet_Converted.yml")
cv2.destroyAllWindows()
