from django.shortcuts import render,redirect
from django.http import HttpResponse

# Create your views here.
def report(request):
    return render(request, "report.html")

def person_report(request):
    return render(request, "person_report.html")

def system_report(request):
    return render(request, "system_report.html")
