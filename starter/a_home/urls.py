from django.urls import path
from a_home.views import *

urlpatterns = [
  path('proyectos/<int:id>', detalle_proyecto, name="detalle_proyecto"),
]

