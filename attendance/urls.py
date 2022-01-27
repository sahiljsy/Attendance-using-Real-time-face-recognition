from django.urls import path
from . import views

urlpatterns = [
    path('report/', views.report),
    path('report/person', views.person_report),
    path('report/system', views.system_report),
]