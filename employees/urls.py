from django.urls import path
from . import views

urlpatterns = [
    path('', views.hello),
    path('signup/', views.signup),
    path('login/', views.Login),
    path('display/', views.Display),
    path('logout/', views.logout_view),
    path('error/404',views.notFound),
    path('updatedetails/',views.Update_details)    
]