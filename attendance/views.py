from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect, StreamingHttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import Http404
import cv2
import dlib
import os
from os.path import isfile, join
import numpy as np
from sklearn.model_selection import train_test_split
from django.contrib import messages
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
from django.http import HttpResponse


@login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def report(request):
    return render(request, "report.html")


@login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def person_report(request):
    return render(request, "person_report.html")


@login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def system_report(request):
    return render(request, "system_report.html")


def checkin(request):
    try:
        data_path = "./data/"
        Training_Data = []
        Labels = []
        img_rows, img_cols = 128, 128
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
        model = load_model("cnn.h5")
        cap = cv2.VideoCapture(0)
        print("Camera opened")
        hogFaceDetector = dlib.get_frontal_face_detector()

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

                ynew = model.predict(inp)
                classes_x = np.argmax(ynew, axis=1)
                color = (255,0, 0)
                text = "Hello, " + Encoded_labels[classes_x[0]] + " Successfully checked in."
                cv2.putText(frame, text,
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, 2)
                cv2.imshow("Face Landmarks", frame)
                # print(ynew[0])
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        print("over")
    except Exception as e:
        cap.release()
        cv2.destroyAllWindows()
        print("Error occured: ", e)
    return redirect("http://127.0.0.1:8000/attendance/markattendance")



def checkout(request):
    try:
        data_path = "./data/"
        Training_Data = []
        Labels = []
        img_rows, img_cols = 128, 128
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
        model = load_model("cnn.h5")
        cap = cv2.VideoCapture(0)
        print("Camera opened")
        hogFaceDetector = dlib.get_frontal_face_detector()

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

                ynew = model.predict(inp)
                classes_x = np.argmax(ynew, axis=1)
                color = (0, 0, 255)
                text = "Hello, " + Encoded_labels[classes_x[0]] + " Successfully checked out."
                cv2.putText(frame, text,
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, 2)
                cv2.imshow("Face Landmarks", frame)
                # print(ynew[0])
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        print("over")
    except Exception as e:
        cap.release()
        cv2.destroyAllWindows()
        print("Error occured: ", e)
    return redirect("http://127.0.0.1:8000/attendance/markattendance")


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
