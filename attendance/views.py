from django.conf import settings
from django.http import HttpResponse
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
import keras
from django.contrib import messages
from sklearn.model_selection import train_test_split
from rsa import verify
import numpy as np
from os.path import isfile, join, exists
from datetime import time, date
import csv
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


def generateTodayReport():
    context = {}
    today = datetime.now()
    dateStr = today.strftime("%d-%b-%y")
    no_user = User.objects.count()
    filename = "attendance_" + \
        today.strftime("%b")+"_"+today.strftime("%y")+".csv"
    if exists(filename) == False:
        context["error"] = "Data is not available"
        return context
    attendance_data = pd.read_csv(filename)
    df = pd.DataFrame(attendance_data, columns=['Username',
                                                'Date',
                                                'Check_in_time',
                                                'Check_out_time',
                                                'Late_check_in',
                                                'early_check_out'])
    present = df.loc[df['Date'] == int(today.strftime("%d"))]
    no_present = present.shape[0]
    no_absent = no_user - no_present
    context['present'] = no_present
    context['absent'] = no_absent
    if no_present != 0:
        late = present.loc[present['Late_check_in'] == True]
        late_emplyoee = late['Username']
        no_late = late.shape[0]
        no_onTime = no_present - no_late
        context['no_late'] = no_late
        context['no_onTime'] = no_onTime

    return context


def generateMonthlyReport(month, username=None):
    month_name = month.split('_')[0]
    month1 = ['Apr', "Jun", 'Sep', 'Nov']
    # month2 = ['Jan', 'Mar', 'May', 'Jul', 'Aug', "Oct", 'Dec']
    context = {}
    context['month'] = month
    filename = "attendance_"+month+".csv"
    if exists(filename) == False:
        context["error"] = "Data is not available"
        return context
    attendance_data = pd.read_csv(filename)
    if month_name == 'Feb':
        total_month_day = np.arange(1, 29)
        working_hour = np.zeros(29)
        no_of_emplyoee_Per_date = np.zeros(29)
        no_of_employee_leave_early = np.zeros(29)
    elif month_name in month1:
        total_month_day = np.arange(1, 31)
        working_hour = np.zeros(30)
        no_of_emplyoee_Per_date = np.zeros(30)
        no_of_employee_leave_early = np.zeros(30)
    else:
        total_month_day = np.arange(1, 32)
        working_hour = np.zeros(31)
        no_of_emplyoee_Per_date = np.zeros(31)
        no_of_employee_leave_early = np.zeros(31)

    if username:
        today = datetime.now()
        print( (str(attendance_data["Date"]) !=  today.strftime("%d")))
        data = attendance_data.loc[(attendance_data['Username'] == username) & (str(attendance_data["Date"]) !=  today.strftime("%d"))]
        print(data)
        checkIn = data['Check_in_time'].to_list()
        checkOut = data['Check_out_time'].to_list()
        date = data['Date'].to_list()
        # date = data['Date']
        # print(checkOut[133])
        if len(date) == 0:
            context["error"] = "Data is not available"
            return context
        print(len(date))
        print(date)
        holidays = set(total_month_day) - set(date)
        if len(date) == 1:
            print("if")
            working_hour[date-1] = ((datetime.strptime(checkOut[0], '%H:%M:%S') -
                                     datetime.strptime(checkIn[0], '%H:%M:%S')).seconds)/3600
        else:
            print("else")
            print(date[3])
            for d in range(len(date)):
                working_hour[date[d]-1] = ((datetime.strptime(checkOut[d], '%H:%M:%S') -
                                            datetime.strptime(checkIn[d], '%H:%M:%S')).seconds)/3600
        context['date'] = total_month_day
        context['working_hour'] = working_hour

    else:
        unique_date = np.unique(attendance_data['Date'])
        holidays = set(total_month_day) - set(unique_date)
        for d in unique_date:
            print(d)
            date_data = attendance_data.loc[attendance_data['Date'] == d]
            eraly_leave = date_data.loc[date_data['early_check_out'] == True]
            no_of_emplyoee_Per_date[d-1] = date_data.shape[0]
            no_of_employee_leave_early[d-1] = eraly_leave.shape[0]
        context["no_user"] = User.objects.count()
        context['date'] = total_month_day
        context['no_of_emplyoee_Per_date'] = no_of_emplyoee_Per_date
        context['no_of_employee_leave_early'] = no_of_employee_leave_early
    print(context)
    return context


def get_csv():
    prefixed = [filename for filename in os.listdir(
        '.') if filename.startswith("attendance_")]
    month = []
    year = []
    for file in prefixed:
        month.append(file.split('_')[1])
        year.append(file.split('_')[2].split('.')[0])
    available_csv = []
    for i in range(len(month)):
        available_csv.append(str(month[i])+"_"+str(year[i]))
    return available_csv


@ login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def report(request):
    return render(request, "report.html")


@ login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def person_report(request):
    available_csv = get_csv()
    today = datetime.now()
    filename = today.strftime("%b")+"_"+today.strftime("%y")
    if request.method == 'POST':
        filename = request.POST["file_name"]
        print("filename:", filename)
        context = generateMonthlyReport(filename, request.user.username)
    else:
        context = generateMonthlyReport(filename, request.user.username)
    context['csv'] = available_csv
    return render(request, "person_report.html", context=context)


@ login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def system_report(request):
    available_csv = get_csv()
    if request.method == 'POST':
        data = request.POST["file_name"]
        print("data:", data)
        if data == "":
            context = generateTodayReport()
        else:
            context = generateMonthlyReport(data)
            context['monthly'] = True

    else:
        context = generateTodayReport()
        context['monthly'] = False
    context['csv'] = available_csv
    print(context)
    return render(request, "system_report.html", context=context)

# def markmyAttendanceIn(name):
#     if exists('attendance.csv') == False:
#         print("if")
#         with open('attendance.csv', 'w+', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(
#                 ["Username", "Date", "Check_in_time", "Check_out_time"])
#             now = datetime.now()
#             dateString = now.strftime('%d-%b-%y')
#             timeString = now.strftime('%H:%M:%S')
#             writer.writerow([name, dateString, timeString])
#     else:
#         print("else")
#         with open('attendance.csv', 'r+', newline='') as file:
#             reader = [row for row in csv.DictReader(file)]
#             now = datetime.now()
#             dateString = now.strftime('%d-%b-%y')
#             timeString = now.strftime('%H:%M:%S')
#             flag = True
#             for row in reader:
#                 if row['Username'] == name and row['Date'] == dateString:
#                     flag = False
#             if flag:
#                 print("new")
#                 file.writelines(f'{name},{dateString},{timeString}')


def markmyAttendanceIn(request, name):
    now = datetime.now()
    # dateString = now.strftime('%d-%b-%y')
    timeString = now.strftime('%H:%M:%S')
    file_name = 'attendance_'+now.strftime('%b_%y')+'.csv'
    entry_time = time(10, 15, 0)
    entry_time = datetime.strptime(str(entry_time), '%H:%M:%S')
    late = datetime.strptime(str(timeString), '%H:%M:%S') > entry_time
    date = now.strftime('%d')
    if exists(file_name) == False:
        print("if")
        with open(file_name, 'w+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Username", "Date", "Check_in_time",
                            "Check_out_time", 'Late_check_in', 'early_check_out'])
            writer.writerow([name, date, timeString, None, late])
            messages.add_message(request, 25, name + ', you have successfully checked in.')
    else:
        print("else")
        with open(file_name, 'r+', newline='') as file:
            reader = [row for row in csv.DictReader(file)]
            writer = csv.writer(file)
            # data = file.readlines()
            flag = True
            for row in reader:
                if row['Username'] == name and row['Date'] == date:
                    flag = False
            if flag:
                print("new")
                writer.writerow([name, date, timeString, None, late])
                messages.add_message(request, 25, name + ', you have successfully checked in.')
            else:
                messages.add_message(request, 25, name + ', you have already checked in.')


def update(header, data, filename):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        writer.writerows(data)


def markmyAttendanceOut(request, name):
    now = datetime.now()
    # dateString = now.strftime('%d-%b-%y')
    timeString = now.strftime('%H:%M:%S')
    out_time = time(7, 0, 0)
    out_time = datetime.strptime(str(out_time), '%H:%M:%S')
    early = datetime.strptime(str(timeString), '%H:%M:%S') < out_time
    date = now.strftime('%d')
    file_name = 'attendance_'+now.strftime('%b_%y')+'.csv'

    if exists(file_name):
        with open(file_name, newline='') as file:
            reader = [row for row in csv.DictReader(file)]
            if reader:
                readHeader = reader[0].keys()
                flag = False

                for row in reader:
                    if row['Username'] == name and row['Date'] == date and row['Check_in_time'] != None and row['Check_out_time'] == '':
                        flag = True
                        row['Check_out_time'] = timeString
                        row['early_check_out'] = early
                print(flag)
                if flag:
                    update(readHeader, reader, file_name)
                    messages.add_message(
                        request, 25, request.user.username + ', you have successfully checked Out.')
                else:
                    messages.add_message(
                        request, 25, request.user.username + ', you have already checked Out or haven\'t checked in.')
            else:
                messages.add_message(
                    request, 25, request.user.username + ', First checked in.')
    else:
        messages.add_message(
            request, 25, request.user.username + ', First checked in.')


# def markmyAttendanceOut(name):
#     if exists('attendance.csv'):
#         with open('attendance.csv', newline='') as file:
#             reader = [row for row in csv.DictReader(file)]
#             readHeader = reader[0].keys()
#             flag = False
#             now = datetime.now()
#             dateString = now.strftime('%d-%b-%y')
#             timeString = now.strftime('%H:%M:%S')
#             for row in reader:
#                 if row['Username'] == name and row['Date'] == dateString and row['Check_in_time'] != None and row['Check_out_time'] == None:
#                     flag = True
#                     row['Check_out_time'] = timeString
#             print(flag)
#             if flag:
#                 update(readHeader, reader, 'attendance.csv')
#             else:
#                 print("First cehck in or You have already Check out")
#     else:
#         print("First cehck in")


def generate_Labels():
    try:
        data_path = "./data/"
        Training_Data = []
        Labels = []
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
        print(Enc_labels)
        return Encoded_labels
    except Exception as e:
        print("Error occured: ", e)
        return []


def open_camera(Encoded_labels, img_rows, img_cols):
    try:
        print("IN")
        verify = []
        print("IN")
        cap = cv2.VideoCapture(0)
        print("IN")
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
        print(ynew)
        best_prediction = max(set(verify), key=verify.count)
        return best_prediction
    except Exception as e:
        cap.release()
        cv2.destroyAllWindows()
        print("Error occured: ", e)


@ login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def checkin(request):
    try:
        data_path = "./data/"
        img_rows, img_cols = 128, 128
        Encoded_labels = generate_Labels()
        best_prediction = open_camera(Encoded_labels, img_rows, img_cols)
        print("System Predicted: ", best_prediction)
        if(request.user.username == best_prediction):
            print("Valid")
            markmyAttendanceIn(request, best_prediction)
        else:
            print("Invalid")
            messages.add_message(
                request, 25, 'Sorry, something went wrong while checking in.')

        print("over")
    except Exception as e:
        print("Error occured: ", e)
    return redirect("http://127.0.0.1:8000/attendance/markattendance")


@ login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def checkout(request):
    try:
        img_rows, img_cols = 128, 128
        Encoded_labels = generate_Labels()
        best_prediction = open_camera(Encoded_labels, img_rows, img_cols)
        print("System Predicted: ", best_prediction)
        if(request.user.username == best_prediction):
            print("Valid")
            markmyAttendanceOut(request, best_prediction)
            # if flag == True:
            #     messages.add_message(
            #         request, 25, best_prediction + ', you have successfully checked-out.')
            # else:
            #     messages.add_message(
            #         request, 25, best_prediction + ', Please check-in first.')
        else:
            print("Invalid")
            messages.add_message(
                request, 25, 'Sorry, something went wrong while checking out.')

        print("over")
    except Exception as e:
        print("Error occured: ", e)
    return redirect("http://127.0.0.1:8000/attendance/markattendance")


@ login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def mark_attendance(request):
    check = settings.IS_TRAINING
    if check == False:
        return render(request, "mark_attendance.html")
    else:
        messages.add_message(
                request, 25, 'Sorry, right now system is getting trained. Please wait...')
        return redirect("http://127.0.0.1:8000/")


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


@ login_required(login_url="http://127.0.0.1:8000/accounts/login/")
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

def returnRes(request):
    print("IN")
    return redirect("http://127.0.0.1:8000/eheth")

@ login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def trainmodel(request):
    try:
        settings.IS_TRAINING = True
        print(settings.IS_TRAINING)
        data_path = "./data/"
        Training_Data = []
        Labels = []
        onlyfiles = [f for f in os.listdir(
            data_path) if isfile(join(data_path, f))]
        # print(onlyfiles)

        for i, files in enumerate(onlyfiles):
            image_path = data_path + onlyfiles[i]
            images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            name = onlyfiles[i].split("_")
            name.pop()
            Labels.append('_'.join(name))
            # Labels.append(onlyfiles[i].split("_")[0])
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
        model.add(Conv2D(filters=40, kernel_size=(3, 3), padding='Same', activation='relu', input_shape=(128, 128, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=60, kernel_size=(2, 2), padding='Same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=160, kernel_size=(3, 3), padding='Same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(no_of_labels, activation="softmax"))
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        model.summary()
        model.fit(x=x_train, y=y_train, epochs=20)

        print("\nTesting Phase:\n\n")
        scor = model.evaluate(np.array(x_test),  np.array(y_test))
        print('test los {:.4f}'.format(scor[0]))
        print('test acc {:.4f}'.format(scor[1]))
        model.save("cnn.h5")        
    except Exception as e:
        print("Error occured: ", e)
    settings.IS_TRAINING = False
    return redirect("http://127.0.0.1:8000/")
