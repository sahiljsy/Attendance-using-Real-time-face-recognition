from django.urls import path
from . import views

urlpatterns = [
    path('report/', views.report),
    path('report/person', views.person_report),
    path('report/employee', views.employee_report),
    path('report/system', views.system_report),
    path('takedata', views.take_data),
    path('markattendance', views.mark_attendance),
    path('checkin', views.checkin),
    path('checkout', views.checkout),
    path('train', views.trainmodel),    
]