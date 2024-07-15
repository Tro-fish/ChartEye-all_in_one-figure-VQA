from django.urls import path
from . import views

urlpatterns = [
    path('', views.extract, name='extract'),
]