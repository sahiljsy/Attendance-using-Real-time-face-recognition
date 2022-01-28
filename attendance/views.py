from django.shortcuts import render,redirect
from django.http import HttpResponse

# Create your views here.
def report(request):
    return render(request, "report.html")

def person_report(request):
    return render(request, "person_report.html")

def system_report(request):
    return render(request, "system_report.html")

def mark_attendance(request):
    return render(request, "mark_attendance.html")

def take_data(request):
    if request.method == 'POST':
        username = request.POST['username']

        if username == "":
            context = {'error': "Username required."}
            return render(request, 'takedata.html', context)

        if User.objects.filter(username=username).exists():
            return render(request, "takedata.html")
        else:
            return render(request, "takedata.html")
    else:
        return render(request, "takedata.html")
