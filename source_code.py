import cv2

face_classifier = cv2.CascadeClassifier("E:\\mini-project21\\haarcascade_frontalface_default.xml")

smile_classifier = cv2.CascadeClassifier("E:\\mini-project21\\haarcascade_smile.xml")

cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    convert_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(convert_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:

        roi_gray = convert_gray[y:y+h, x:x+w]

        roi_color = frame[y:y+h, x:x+w]

        smiles = smile_classifier.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=30, minSize=(15, 15))

        for (sx, sy, sw, sh) in smiles:
            if (sx, sy, sw, sh):
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
                cv2.putText(frame, "Mask not Detected", (200, 450),cv2.FONT_HERSHEY_PLAIN, 2.3, (255, 255, 0), 2)
        cv2.putText(frame, "Mask Detection", (10, 25),cv2.FONT_HERSHEY_PLAIN, 1.0, (150, 150, 0), 2)
        
    cv2.imshow("My video", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()

cap.deleteAllWindows()