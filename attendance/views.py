from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect, StreamingHttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import Http404
import cv2
import dlib
import threading
import os


@login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def report(request):
    return render(request, "report.html")


@login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def person_report(request):
    return render(request, "person_report.html")


@login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def system_report(request):
    return render(request, "system_report.html")


def mark_attendance(request):
    return render(request, "mark_attendance.html")


def create_dataset(username):
    try:
        if(os.path.exists('./data') == False):
            os.makedirs('./data')
        cap = cv2.VideoCapture(0)
        print("Camera opened")
        hogFaceDetector = dlib.get_frontal_face_detector()
        skip = 0
        while True:
            _, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = hogFaceDetector(gray, 1)
            for (i, rect) in enumerate(faces):
                x = rect.left()
                y = rect.top()
                w = rect.right() - x
                h = rect.bottom() - y
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_cropped = frame[y-5:y+h-5, x-5:x+w-5]
                face_cropped = cv2.resize(face_cropped, (64, 64))
                face_cropped = cv2.cvtColor(face_cropped, cv2.COLOR_BGR2GRAY)
                skip = skip+1
                if skip % 10 == 0:
                    file_path = "./data/"+username + \
                        "_"+str(int(skip/10))+".jpg"
                    cv2.imwrite(file_path, face_cropped)
            # draw a rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Face Landmarks", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or skip == 150:
                break
        cap.release()
        cv2.destroyAllWindows()
        print("Working")
    except Exception as e:
        cap.release()
        cv2.destroyAllWindows()
        print("Error occured: ", e)
    return


def take_data(request):
    if not request.user.is_staff:
        return redirect("http://127.0.0.1:8000/error/404")
    if request.method == 'POST':
        username = request.POST['username']

        if username == "":
            context = {'error': "Username required."}
            return render(request, 'takedata.html', context)

        if User.objects.filter(username=username).exists():
            create_dataset(username)
            return render(request, "takedata.html")
        else:
            context = {'error': "Username not find."}
            return render(request, "takedata.html", context)
    else:
        return render(request, "takedata.html")
