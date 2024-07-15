from django.contrib import admin
from a_home.models import Clave, Desarrollador, Proyecto, Repositorio, Servidor

admin.site.register(Proyecto)
admin.site.register(Desarrollador)
admin.site.register(Repositorio)
admin.site.register(Servidor)
admin.site.register(Clave)