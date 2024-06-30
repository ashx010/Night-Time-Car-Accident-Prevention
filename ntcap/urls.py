# object_detection_site/urls.py
from django.contrib import admin
from django.urls import path
from detection import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('index', views.index, name='index'),
    path('', views.first, name='first'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('toggle_pause/', views.toggle_pause, name='toggle_pause'),
    path('get_log/', views.get_log, name='get_log'),
    path('update_config/', views.update_config, name='update_config'),
]
