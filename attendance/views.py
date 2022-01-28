from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect,StreamingHttpResponse
from django.contrib.auth.decorators import login_required
import cv2
import threading
# import imutils
# from imutils.video import VideoStream
# from imutils import face_utils


@login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def report(request):
    return render(request, "report.html")


@login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def person_report(request):
    return render(request, "person_report.html")


@login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def system_report(request):
    return render(request, "system_report.html")


def create_dataset(request):
    try:
        cap = cv2.VideoCapture(0)
        print("Camera opened")
        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        print("xml loaded")
        skip = 0 
        while True:
            ret, frame = cap.read()
            if ret == False:
                continue
            cv2.imshow("Face Section", frame)
            faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        #     if len(faces) > 0:
        #         for (x,y,w,h) in faces:
        #             face_cropped = frame[y-10:y+h+10, x-10:x+w+10]
        #             frame = face_cropped
        #             face_cropped = cv2.resize(face_cropped, (450,450))
        #             face_cropped = cv2.cvtColor(face_cropped, cv2.COLOR_BGR2GRAY)
        #             skip = skip+1
        #         # if skip % 10 == 0:
        #         #     file_path = "./data/sahil_"+str(skip)+".jpg"
        #         #     cv2.imwrite(file_path, face_cropped)
        # #cv2.putText(face_cropped,str(skip), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255,255), 2)
        # #cv2.imshow("Vedio Frame", frame)
        #         cv2.imshow("Face Section", face_cropped)
            skip = skip+2
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        print("Working")
    except Exception as e:
        cap.release()
        cv2.destroyAllWindows()
        print("Error occured: ", e)
    return redirect("http://127.0.0.1:8000/")
   



