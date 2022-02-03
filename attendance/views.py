from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect, StreamingHttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import Http404
import cv2
import threading
import os
from os.path import isfile, join
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics


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
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("xml loaded")
        skip = 0
        while True:
            ret, frame = cap.read()
            if ret == False:
                continue
            cv2.imshow("Face Section", frame)
            faces = face_cascade.detectMultiScale(frame, 1.3, 5)
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face_cropped = frame[y-10:y+h+10, x-10:x+w+10]
                    frame = face_cropped
                    face_cropped = cv2.resize(face_cropped, (128, 128))
                    face_cropped = cv2.cvtColor(
                        face_cropped, cv2.COLOR_BGR2GRAY)
                    skip = skip+1
                    if skip % 10 == 0:
                        file_path = "./data/"+username + \
                            "_"+str(int(skip/10))+".jpg"
                        cv2.imwrite(file_path, face_cropped)
                    cv2.imshow("Face Section", face_cropped)
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


def trainmodel(request):
    data_path = "./data/"
    Training_Data = []
    Labels = []
    onlyfiles = [f for f in os.listdir(data_path) if isfile(join(data_path, f))]

    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Labels.append(onlyfiles[i].split("_")[0])
        Training_Data.append(np.asarray(images, dtype=np.uint8))

    print(np.asarray(Training_Data))
    lbl_en = LabelEncoder()
    encoded_lbls = lbl_en.fit_transform(Labels)
    print(encoded_lbls)

    unique_lbls = np.unique(np.array(encoded_lbls))
    # print(unique_lbls)
    data_train, data_test, target_train, target_test = train_test_split(np.asarray(Training_Data),encoded_lbls, test_size = 0.20, random_state = 58)
    nsamples, nx, ny = data_train.shape
    d2_train_dataset = data_train.reshape((nsamples,nx*ny))
    nsamples, nx, ny = data_test.shape
    d2_test_dataset = data_test.reshape((nsamples,nx*ny))
    gnb = GaussianNB()
    gnb.fit(d2_train_dataset, target_train)
    target_pred = gnb.predict(d2_test_dataset)
    print("Accuracy:",metrics.accuracy_score(target_test, target_pred))
    precision = metrics.precision_score(target_test, target_pred, average=None)
    recall = metrics.recall_score(target_test, target_pred, average=None)
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    return redirect("http://127.0.0.1:8000/")
