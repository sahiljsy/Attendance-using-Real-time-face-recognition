from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.contrib.auth import login, authenticate, logout
from .forms import SignUpForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required

@login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def hello(request):
        return render(request, "home.html")

def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect("http://127.0.0.1:8000/accounts/login/")
    else:
        form = SignUpForm()
    return render(request, 'signup.html', {'form': form})

def Login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username = username, password = password)
        if user is not None:
            form = login(request, user)
            return redirect("http://127.0.0.1:8000/")
    form = AuthenticationForm()
    return render(request, 'login.html', {'form':form})

def logout_view(request):
    logout(request)
    return redirect('http://127.0.0.1:8000/accounts/login')

@login_required(login_url="http://127.0.0.1:8000/accounts/login/")
def Display(request):
    return render(request, "display.html")
