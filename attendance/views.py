from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect, StreamingHttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import Http404
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import matplotlib.dates as mdates
import cv2
import dlib
import os
import csv
from os.path import isfile, join, exists
import numpy as np
from rsa import verify
from sklearn.model_selection import train_test_split
from django.contrib import messages
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from django.http import HttpResponse
from django.conf import settings


@login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def report(request):
    return render(request, "report.html")


@login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def person_report(request):
    return render(request, "person_report.html")


@login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def system_report(request):
    return render(request, "system_report.html")


def markmyAttendanceIn(name):
    if exists('attendance.csv') == False:
        print("if")
        with open('attendance.csv', 'w+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Username", "Date", "Check_in_time", "Check_out_time"])
            now = datetime.now()
            dateString = now.strftime('%d-%b-%y')
            timeString = now.strftime('%H:%M:%S')
            writer.writerow([name, dateString, timeString])
    else:
        print("else")
        with open('attendance.csv', 'r+', newline='') as file:
            reader = [row for row in csv.DictReader(file)]
            now = datetime.now()
            dateString = now.strftime('%d-%b-%y')
            timeString = now.strftime('%H:%M:%S')
            flag = True
            for row in reader:
                if row['Username'] == name and row['Date'] == dateString:
                    flag = False
            if flag:
                print("new")
                file.writelines(f'{name},{dateString},{timeString}')


def update(header, data, filename):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        writer.writerows(data)


def markmyAttendanceOut(name):
    if exists('attendance.csv'):
        with open('attendance.csv', newline='') as file:
            reader = [row for row in csv.DictReader(file)]
            readHeader = reader[0].keys()
            flag = False
            now = datetime.now()
            dateString = now.strftime('%d-%b-%y')
            timeString = now.strftime('%H:%M:%S')
            for row in reader:
                if row['Username'] == name and row['Date'] == dateString and row['Check_in_time'] != None and row['Check_out_time'] == None:
                    flag = True
                    row['Check_out_time'] = timeString
            print(flag)
            if flag:
                update(readHeader, reader, 'attendance.csv')
            else:
                print("First cehck in or You have already Check out")
    else:
        print("First cehck in")


@login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def checkin(request):
    try:
        data_path = "./data/"
        Training_Data = []
        Labels = []
        verify = []
        img_rows, img_cols = 128, 128
        onlyfiles = [f for f in os.listdir(
            data_path) if isfile(join(data_path, f))]

        for i, files in enumerate(onlyfiles):
            image_path = data_path + onlyfiles[i]
            images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            name = onlyfiles[i].split("_")
            name.pop()
            Labels.append('_'.join(name))
            Labels.append(onlyfiles[i].split("_")[0])
            Training_Data.append(np.asarray(images, dtype=np.uint8))

        Encoded_labels = []
        for i in Labels:
            if i not in Encoded_labels:
                Encoded_labels.append(i)

        Encoded_dict = {}
        for i in range(len(Encoded_labels)):
            Encoded_dict[Encoded_labels[i]] = i
        print(Encoded_labels, Encoded_dict)

        Enc_labels = []
        for i in Labels:
            Enc_labels.append(Encoded_dict[i])
        no_of_labels = len(Encoded_labels)
        cap = cv2.VideoCapture(0)
        print("Camera opened")
        hogFaceDetector = dlib.get_frontal_face_detector()
        cnt = 0
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
                face_cropped = frame[y:y+h, x:x+w]
                face_cropped = cv2.resize(face_cropped, (128, 128))
                face_cropped = cv2.cvtColor(face_cropped, cv2.COLOR_BGR2GRAY)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_cropped = (face_cropped/255)
                inp = face_cropped.reshape(1, img_rows, img_cols, 1)
                ynew = settings.MODEL.predict(inp)
                classes_x = np.argmax(ynew, axis=1)
                verify.append(Encoded_labels[classes_x[0]])
                color = (0, 0, 255)
                text = "Hello user, Please see towards camera with"
                cv2.putText(frame, text,
                            (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, 2)
                text = "proper lighting for proper face recognition."
                cv2.putText(frame, text,
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, 2)
                cnt = cnt + 1
            cv2.imshow("Face Landmarks", frame)
            key = cv2.waitKey(1) & 0xFF
            if cnt == 30:  # key == ord('q') or
                break
        cap.release()
        cv2.destroyAllWindows()
        best_prediction = max(set(verify), key=verify.count)
        print(set(verify))
        if(request.user.username == best_prediction):
            print("Valid")
            markmyAttendanceIn(best_prediction)
            messages.add_message(
                request, 25, best_prediction + ', you have successfully checked in.')
        else:
            print("Invalid")
            messages.add_message(
                request, 25, 'Sorry, something went wrong while checking in.')

        print("over")
    except Exception as e:
        cap.release()
        cv2.destroyAllWindows()
        print("Error occured: ", e)
    return redirect("http://127.0.0.1:8000/attendance/markattendance")


@login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def checkout(request):
    try:
        data_path = "./data/"
        Training_Data = []
        Labels = []
        verify = []
        img_rows, img_cols = 128, 128
        onlyfiles = [f for f in os.listdir(
            data_path) if isfile(join(data_path, f))]

        for i, files in enumerate(onlyfiles):
            image_path = data_path + onlyfiles[i]
            images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            name = onlyfiles[i].split("_")
            name.pop()
            Labels.append('_'.join(name))
            Labels.append(onlyfiles[i].split("_")[0])
            Training_Data.append(np.asarray(images, dtype=np.uint8))

        Encoded_labels = []
        for i in Labels:
            if i not in Encoded_labels:
                Encoded_labels.append(i)

        Encoded_dict = {}
        for i in range(len(Encoded_labels)):
            Encoded_dict[Encoded_labels[i]] = i
        print(Encoded_labels, Encoded_dict)

        Enc_labels = []
        for i in Labels:
            Enc_labels.append(Encoded_dict[i])
        no_of_labels = len(Encoded_labels)
        cap = cv2.VideoCapture(0)
        print("Camera opened")
        hogFaceDetector = dlib.get_frontal_face_detector()
        cnt = 0
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
                face_cropped = frame[y:y+h, x:x+w]
                face_cropped = cv2.resize(face_cropped, (128, 128))
                face_cropped = cv2.cvtColor(face_cropped, cv2.COLOR_BGR2GRAY)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_cropped = (face_cropped/255)
                inp = face_cropped.reshape(1, img_rows, img_cols, 1)
                ynew = settings.MODEL.predict(inp)
                classes_x = np.argmax(ynew, axis=1)
                verify.append(Encoded_labels[classes_x[0]])
                color = (0, 0, 255)
                text = "Hello user, Please see towards camera with"
                cv2.putText(frame, text,
                            (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, 2)
                text = "proper lighting for proper face recognition."
                cv2.putText(frame, text,
                            (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, 2)
                cnt = cnt + 1
            cv2.imshow("Face Landmarks", frame)
            key = cv2.waitKey(1) & 0xFF
            if cnt == 30:  # key == ord('q') or
                break
        cap.release()
        cv2.destroyAllWindows()
        best_prediction = max(set(verify), key=verify.count)
        print(verify)
        if(request.user.username == best_prediction):
            print("Valid")
            markmyAttendanceOut(best_prediction)
            messages.add_message(
                request, 25, best_prediction + ', you have successfully checked out.')
        else:
            print("Invalid")
            messages.add_message(
                request, 25, 'Sorry, something went wrong while checking out.')

        print("over")
    except Exception as e:
        cap.release()
        cv2.destroyAllWindows()
        print("Error occured: ", e)
    return redirect("http://127.0.0.1:8000/attendance/markattendance")


@login_required(login_url="http://127.0.0.1:8000/accounts/login/")
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
                face_cropped = cv2.resize(face_cropped, (128, 128))
                face_cropped = cv2.cvtColor(face_cropped, cv2.COLOR_BGR2GRAY)
                skip = skip+1
                file_path = "./data/"+username + \
                    "_"+str(int(skip))+".jpg"
                cv2.imwrite(file_path, face_cropped)
                # draw a rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Face Landmarks", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or skip == 100:
                break
        cap.release()
        cv2.destroyAllWindows()
        print("Working")

    except Exception as e:
        cap.release()
        cv2.destroyAllWindows()
        print("Error occured: ", e)
    return


@login_required(login_url="http://127.0.0.1:8000/accounts/login/")
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


@login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def trainmodel(request):
    try:
        data_path = "./data/"
        Training_Data = []
        Labels = []
        onlyfiles = [f for f in os.listdir(
            data_path) if isfile(join(data_path, f))]
        # print(onlyfiles)

        for i, files in enumerate(onlyfiles):
            image_path = data_path + onlyfiles[i]
            images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            Labels.append(onlyfiles[i].split("_")[0])
            Training_Data.append(np.asarray(images, dtype=np.uint8))

        Encoded_labels = []
        for i in Labels:
            if i not in Encoded_labels:
                Encoded_labels.append(i)

        Encoded_dict = {}
        for i in range(len(Encoded_labels)):
            Encoded_dict[Encoded_labels[i]] = i
        print(Encoded_labels, Encoded_dict)

        Enc_labels = []
        for i in Labels:
            Enc_labels.append(Encoded_dict[i])
        no_of_labels = len(Encoded_labels)
        print(no_of_labels)
        print(np.asarray(Training_Data).shape, np.array(Enc_labels).shape)
        x_train, x_test, y_train, y_test = train_test_split(
            np.asarray(Training_Data), np.asarray(Enc_labels), test_size=0.3)
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        img_rows, img_cols = 128, 128

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train = x_train/255
        x_test = x_test/255

        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

        model = Sequential()
        model.add(Conv2D(filters=40, kernel_size=(3, 3), padding='Same',
                         activation='relu', input_shape=(128, 128, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=60, kernel_size=(2, 2),
                         padding='Same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=160, kernel_size=(
            3, 3), padding='Same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(no_of_labels, activation="softmax"))
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        model.fit(x=x_train, y=y_train, epochs=20)

        print("\nTesting Phase:\n\n")
        scor = model.evaluate(np.array(x_test),  np.array(y_test))
        print('test los {:.4f}'.format(scor[0]))
        print('test acc {:.4f}'.format(scor[1]))
        model.save("cnn.h5")
    except Exception as e:
        print("Error occured: ", e)
    return redirect("http://127.0.0.1:8000/")
