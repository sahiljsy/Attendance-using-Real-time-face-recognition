from django.urls import path
from . import views

urlpatterns = [
    path('', views.hello),
    path('signup/', views.signup),
    path('login/', views.Login)
]