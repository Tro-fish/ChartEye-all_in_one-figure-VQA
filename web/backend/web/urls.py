from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('extract/', include('extract_images.urls')),
    path('caption/', include('caption.urls')),
    path('chat/', include('chat.urls')),
    path('admin/', admin.site.urls),
]
