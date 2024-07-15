from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('extract/', include('extract_figure.urls')),
    path('admin/', admin.site.urls),
]
