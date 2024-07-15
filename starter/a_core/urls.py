"""
URL configuration for a_core project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from a_users.views import profile_view
from a_home.views import *

urlpatterns = [
    path('admin/', admin.site.urls), # admin panel
    path('accounts/', include('allauth.urls')), # lib allauth
    # urls para el view de home:
    path('', home_view, name="home"),
    # importación urls de app a_users:
    path('profile/', include('a_users.urls')),
    # importación urls de app a_home:
    path('', include('a_home.urls')),
    # view de perfil:
    path('@<username>/', profile_view, name="profile"),
    # importación urls de app a_invoices:
    path('', include('a_invoices.urls')),
]

# Only used when DEBUG=True, whitenoise can serve files when DEBUG=False
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
