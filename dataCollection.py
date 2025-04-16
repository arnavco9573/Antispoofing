import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
from time import time

################################################
classID = 0 #0 is fake 1 is real
blurThreshold = 35 #Larger is more focused
outputFolderPath = 'Datasets/DataCollect'
confidence = 0.8
save=True

debug = False
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint = 6

###############################################
cap = cv2.VideoCapture(0)
cap.set(3,camWidth)
cap.set(4,camHeight)
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

while True:
    success, img = cap.read()
    imgOut = img.copy()
    img, bboxs = detector.findFaces(img,draw=False)

    listBlur = []  # True False Values indication if the faces are blur or not
    listInfo = []  # The normalized values and the class name for the label text file

    if bboxs:
        for bbox in bboxs:
            x,y,w,h = bbox['bbox']
            score = bbox['score'][0]
            # print(score)
            # print(x,y,w,h)

            # Check the score
            if score > confidence:
                # Adding an offset to the face Detected
                offsetW = (offsetPercentageW/100)*w
                x = int(x-offsetW)
                w = int(w + offsetW * 2)

                offsetH = (offsetPercentageH/100)*h
                y = int(y-offsetH*3)
                h = int(h + offsetH * 3.5)

                # To avoid values below 0
                if x<0:x=0
                if y<0:y=0
                if w>img.shape[1]:w=img.shape[1]
                if h>img.shape[0]:h=img.shape[0]

                # Find the Blurriness of the face
                imgFace = img[y:y+h,x:x+w]
                cv2.imshow('Face', imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())

                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)

                # Normalize values
                ih,iw,_ = img.shape
                # center point values
                xc,yc = x+w/2,y+h/2
                # center point normalize values
                xcn,ycn = round(xc/iw,floatingPoint),round(yc/ih,floatingPoint)
                wn,hn = round(w/iw,floatingPoint),round(h/ih,floatingPoint)
                # print(xcn,ycn,wn,hn)

                # Avoid values above 1
                if xcn>1:xcn=1
                if ycn<1:ycn=1
                if wn>1:wn=1
                if hn>1:hn=1

                # Format in which YOLO requires the label
                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}")

                # Drawing the images
                cv2.rectangle(imgOut, (x, y, w, h), (255, 0, 0), 2)
                cvzone.putTextRect(imgOut, f'Score:{int(score * 100)}% Blur:{blurValue}', (x, y - 20), scale=2,
                                   thickness=3)

                if debug:
                    cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 2)
                    cvzone.putTextRect(img, f'Score:{int(score * 100)}% Blur:{blurValue}', (x, y - 20), scale=2,
                                       thickness=3)


    # To save images and text file
    if save:
        if all(listBlur) and listBlur!=[]:
            # Save Image
            timeNow = time()
            timeNow = str(timeNow).split('.')
            timeNow = timeNow[0]+timeNow[1]
            cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg",img)
            #save the label

            for info in listInfo:
                with open(f"{outputFolderPath}/{timeNow}.txt", "a") as f:
                    f.write(info)



    cv2.imshow("Image", imgOut)
    cv2.waitKey(1)
